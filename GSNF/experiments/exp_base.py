import copy
import logging
import random
import sys
import time
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import numpy as np
import sklearn as sk
import torch
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from experiments.data_eicu import get_eicu_tvt_datasets
from experiments.data_mimic4 import collate_fn_biclass, get_mimic4_tvt_datasets
from experiments.data_physionet12 import get_physionet12_tvt_datasets
from experiments.data_physionet import get_physionet_tvt_datasets
from model.model_factory import ModelFactory
from utils import record_experiment


PHYSIONET12_FIXED_RANDOM_STATE = 290
P12_FIXED_RANDOM_STATE = 155


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, seed=1, pos_repeat=3, logger=None):
        if batch_size < 2:
            raise ValueError("balanced batch sampler requires batch_size >= 2")

        label_df = getattr(dataset, "label_df", None)
        if label_df is None or "labels" not in label_df:
            raise ValueError("balanced batch sampler requires label_df['labels']")

        labels = label_df.sort_index()["labels"].astype(float).to_numpy()
        self.pos_idx = np.flatnonzero(labels == 1.0).astype(np.int64)
        self.neg_idx = np.flatnonzero(labels == 0.0).astype(np.int64)
        if len(self.pos_idx) == 0 or len(self.neg_idx) == 0:
            raise ValueError(
                f"balanced batch sampler needs both classes, got "
                f"pos={len(self.pos_idx)}, neg={len(self.neg_idx)}")

        self.batch_size = int(batch_size)
        self.neg_per_batch = self.batch_size // 2
        self.pos_per_batch = self.batch_size - self.neg_per_batch
        self.seed = int(seed)
        self.pos_repeat = max(1, int(pos_repeat))
        self.epoch = 0
        self.n_batches = min(
            len(self.neg_idx) // self.neg_per_batch,
            (len(self.pos_idx) * self.pos_repeat) // self.pos_per_batch)
        self.n_batches = max(1, self.n_batches)

        if logger is not None:
            logger.info(
                "balanced_batch_sampler=True "
                f"batch_size={self.batch_size} "
                f"neg_per_batch={self.neg_per_batch} "
                f"pos_per_batch={self.pos_per_batch} "
                f"pos_repeat={self.pos_repeat} "
                f"n_batches={self.n_batches} "
                f"num_neg={len(self.neg_idx)} "
                f"num_pos={len(self.pos_idx)}")

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        self.epoch += 1

        neg = self.neg_idx.copy()
        pos = np.tile(self.pos_idx, self.pos_repeat)
        rng.shuffle(neg)
        rng.shuffle(pos)

        for batch_idx in range(self.n_batches):
            neg_start = batch_idx * self.neg_per_batch
            pos_start = batch_idx * self.pos_per_batch

            if neg_start + self.neg_per_batch <= len(neg):
                neg_batch = neg[neg_start:neg_start + self.neg_per_batch]
            else:
                neg_batch = rng.choice(self.neg_idx, size=self.neg_per_batch, replace=True)

            if pos_start + self.pos_per_batch <= len(pos):
                pos_batch = pos[pos_start:pos_start + self.pos_per_batch]
            else:
                pos_batch = rng.choice(self.pos_idx, size=self.pos_per_batch, replace=True)

            batch = np.concatenate([neg_batch, pos_batch]).astype(np.int64)
            rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return self.n_batches


class BaseExperiment:
    """Base class for GSNF binary classification experiments."""

    def __init__(self, args: Namespace):
        requested_random_state = getattr(args, "random_state", None)
        if getattr(args, "data", None) == "physionet12":
            args.random_state = PHYSIONET12_FIXED_RANDOM_STATE
        elif getattr(args, "data", None) == "p12":
            args.random_state = P12_FIXED_RANDOM_STATE

        self.args = args
        self.epochs_max = args.epochs_max
        self.patience = args.patience
        self.proj_path = Path(args.proj_path)
        self.data_dir = Path(args.data_dir) if args.data_dir else self.proj_path / "data"
        self.mf = ModelFactory(self.args)
        self.tags = [
            "biclass",
            self.args.data,
            "gsnf",
            self.args.test_info,
        ]
        self.args.exp_name = "_".join(self.tags + [("r" + str(args.random_state))])

        torch.manual_seed(args.random_state)
        np.random.seed(args.random_state)
        random.seed(args.random_state)

        self._init_logger()
        if self.args.data == "physionet12":
            self.logger.info(
                f"physionet12 uses fixed random_state={PHYSIONET12_FIXED_RANDOM_STATE}; "
                f"requested_random_state={requested_random_state}")
        elif self.args.data == "p12":
            self.logger.info(
                f"p12 uses fixed random_state={P12_FIXED_RANDOM_STATE}; "
                f"requested_random_state={requested_random_state}")
        self.logger.info(f"data_dir={self.data_dir}")

        self.device = torch.device(args.device)
        self.logger.info(f"Device: {self.device}")

        self.variable_num, self.dltrain, self.dlval, self.dltest = self.get_data(args)
        self.args.variable_num = self.variable_num
        if self.args.latent_dim != self.variable_num:
            self.logger.info(
                f"GSNF uses latent_dim=variable_num; overriding latent_dim "
                f"from {self.args.latent_dim} to {self.variable_num}")
            self.args.latent_dim = self.variable_num
        if getattr(self.args, "graph_num_nodes", 0) != self.variable_num:
            self.logger.info(f"Setting graph_num_nodes to variable_num={self.variable_num}")
            self.args.graph_num_nodes = self.variable_num

        self._prepare_biclass_tricks()
        self.model = self.get_model().to(self.device)
        if self.args.init_classifier_bias:
            self.model.init_classifier_bias(getattr(self.args, "train_pos_rate", None))
        num_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"num_params={num_params}")

        self.optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        self.scheduler = None
        self.scheduler_type = getattr(args, "lr_scheduler", "step")
        if self.scheduler_type == "step" and args.lr_scheduler_step > 0:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optim, args.lr_scheduler_step, args.lr_decay)
        elif self.scheduler_type == "plateau":
            monitor = getattr(args, "plateau_monitor", "auprc")
            plateau_mode = getattr(args, "plateau_mode", "auto")
            if plateau_mode == "auto":
                plateau_mode = "max" if monitor in ["auprc", "auroc"] else "min"
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optim,
                mode=plateau_mode,
                factor=getattr(args, "plateau_factor", 0.1),
                patience=getattr(args, "plateau_patience", 1),
                threshold=getattr(args, "plateau_threshold", 1e-4),
                threshold_mode="rel",
                min_lr=getattr(args, "plateau_min_lr", 1e-8),
                eps=1e-8,
            )
            self.logger.info(
                f"lr_scheduler=plateau monitor={monitor} mode={plateau_mode} "
                f"factor={getattr(args, 'plateau_factor', 0.1)} "
                f"patience={getattr(args, 'plateau_patience', 1)}")

    @staticmethod
    def _scalar(value):
        if isinstance(value, torch.Tensor):
            return value.detach().item()
        return float(value)

    def _move_batch_to_device(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        if isinstance(batch, dict):
            return {key: self._move_batch_to_device(value) for key, value in batch.items()}
        if isinstance(batch, list):
            return [self._move_batch_to_device(value) for value in batch]
        if isinstance(batch, tuple):
            return tuple(self._move_batch_to_device(value) for value in batch)
        return batch

    def _metric_value(self, metrics, key, default=0.0):
        if not isinstance(metrics, dict) or key not in metrics:
            return default
        return self._scalar(metrics[key])

    def _average_state_dict(self, averaged_state, current_state, num_updates):
        if averaged_state is None:
            return {key: value.detach().clone() for key, value in current_state.items()}

        averaged = {}
        for key, value in current_state.items():
            value = value.detach()
            if torch.is_floating_point(value):
                averaged[key] = averaged_state[key] + (
                    value - averaged_state[key]) / float(num_updates)
            else:
                averaged[key] = value.clone()
        return averaged

    def _update_ema_state_dict(self, ema_state, current_state, decay):
        if ema_state is None:
            return {key: value.detach().clone() for key, value in current_state.items()}

        updated = {}
        for key, value in current_state.items():
            value = value.detach()
            if torch.is_floating_point(value):
                updated[key] = ema_state[key] * decay + value * (1.0 - decay)
            else:
                updated[key] = value.clone()
        return updated

    def _prepare_biclass_tricks(self):
        self.args.train_pos_rate = None
        self.args.ce_pos_weight_runtime = max(
            0.0, float(getattr(self.args, "ce_pos_weight", 0.0)))

        dataset = getattr(self.dltrain, "dataset", None)
        label_df = getattr(dataset, "label_df", None)
        if label_df is not None and "labels" in label_df:
            labels = label_df["labels"].astype(float).to_numpy()
            labels = labels[~np.isnan(labels)]
            if labels.size > 0:
                pos_rate = float(labels.mean())
                self.args.train_pos_rate = pos_rate
                self.logger.info(f"train_pos_rate={pos_rate:.6f}")

        if self.args.balanced_ce and self.args.ce_pos_weight_runtime <= 0.0:
            pos_rate = self.args.train_pos_rate
            if pos_rate is not None and 0.0 < pos_rate < 1.0:
                self.args.ce_pos_weight_runtime = (1.0 - pos_rate) / pos_rate

        if self.args.ce_pos_weight_runtime > 0.0:
            self.logger.info(f"ce_pos_weight={self.args.ce_pos_weight_runtime:.6f}")

    def _format_metrics(self, metrics, keys):
        if not isinstance(metrics, dict):
            return ""
        parts = []
        for key in keys:
            if key in metrics:
                parts.append(f"{key}={self._scalar(metrics[key]):.5f}")
        return " ".join(parts)

    def _summary_metric_keys(self):
        return ["auroc", "auprc", "ce_loss", "loss_itg", "loss_rtg", "graph_kl"]

    def _early_stop_mode(self):
        metric = getattr(self.args, "early_stop_metric", "val_loss")
        if metric in ["auprc", "auroc"]:
            return "max"
        return "min"

    def _is_better(self, value, best, mode):
        if mode == "max":
            return value > best
        return value < best

    def _init_logger(self):
        (self.proj_path / "log").mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=self.proj_path / "log" / (self.args.exp_name + ".log"),
            filemode="w",
            level=logging.INFO,
            force=True,
        )
        self.logger = logging.getLogger()

    def get_model(self):
        print("current_ehr_variable_num: " + str(self.variable_num))
        return self.mf.initialize_biclass_model()

    def get_data(self, args):
        self.args.num_times = self.args.time_max + 1
        if self.args.data == "MIMIC4":
            train_data, val_data, test_data = get_mimic4_tvt_datasets(
                self.args, self.data_dir, self.logger)
        elif self.args.data in ["p12", "P19"]:
            train_data, val_data, test_data = get_physionet_tvt_datasets(
                self.args, self.data_dir, self.logger)
        elif self.args.data == "physionet12":
            train_data, val_data, test_data = get_physionet12_tvt_datasets(
                self.args, self.data_dir, self.logger)
        elif self.args.data == "eICU":
            train_data, val_data, test_data = get_eicu_tvt_datasets(
                self.args, self.data_dir, self.logger)
        else:
            raise ValueError(f"Unsupported dataset: {self.args.data}")

        collate_args = copy.copy(self.args)
        collate_args.device = torch.device("cpu")

        if getattr(self.args, "balanced_batch_sampler", False):
            train_batch_sampler = BalancedBatchSampler(
                train_data,
                self.args.batch_size,
                seed=self.args.random_state,
                pos_repeat=getattr(self.args, "balanced_batch_pos_repeat", 3),
                logger=self.logger,
            )
            dl_train = DataLoader(
                dataset=train_data,
                collate_fn=lambda batch: collate_fn_biclass(
                    batch, train_data.variable_num, collate_args),
                batch_sampler=train_batch_sampler,
                num_workers=self.args.num_dl_workers,
            )
        else:
            dl_train = DataLoader(
                dataset=train_data,
                collate_fn=lambda batch: collate_fn_biclass(
                    batch, train_data.variable_num, collate_args),
                shuffle=True,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_dl_workers,
            )

        dl_val = DataLoader(
            dataset=val_data,
            collate_fn=lambda batch: collate_fn_biclass(
                batch, val_data.variable_num, collate_args),
            shuffle=False,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_dl_workers,
        )
        dl_test = DataLoader(
            dataset=test_data,
            collate_fn=lambda batch: collate_fn_biclass(
                batch, test_data.variable_num, collate_args),
            shuffle=False,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_dl_workers,
        )

        return train_data.variable_num, dl_train, dl_val, dl_test

    def training_step(self, batch):
        batch = self._move_batch_to_device(batch)
        results = self.model.compute_prediction_results(batch)
        return results["loss"]

    def validation_step(self, epoch):
        raise NotImplementedError

    def test_step(self):
        raise NotImplementedError

    def compute_results_all_batches(self, dl):
        total = {
            "loss": 0,
            "likelihood": 0,
            "mse": 0,
            "auroc": 0,
            "auprc": 0,
            "kl_first_p": 0,
            "std_first_p": 0,
            "ce_loss": 0,
            "mse_reg": 0,
            "mae_reg": 0,
            "forward_time": 0,
            "kldiv_z0": 0,
            "loss_ll_z": 0,
            "loss_itg": 0,
            "loss_rtg": 0,
            "graph_kl": 0,
            "val_loss": 0,
            "lat_variance": 0,
        }

        n_test_batches = 0
        classif_predictions = torch.empty(0, device=self.device)
        all_test_labels = torch.empty(0, device=self.device)
        n_traj_samples = self.args.k_iwae

        for batch in dl:
            batch = self._move_batch_to_device(batch)
            results = self.model.run_validation(batch)

            n_labels = 1
            classif_predictions = torch.cat(
                (
                    classif_predictions,
                    results["label_predictions"].reshape(n_traj_samples, -1, n_labels),
                ),
                dim=1,
            )
            all_test_labels = torch.cat(
                (all_test_labels, batch["truth"].reshape(-1, n_labels)),
                dim=0,
            )

            for key in total.keys():
                if results.get(key) is not None:
                    var = results[key]
                    if isinstance(var, torch.Tensor):
                        var = var.detach()
                    total[key] += var
            n_test_batches += 1

        if n_test_batches > 0:
            for key in total.keys():
                total[key] = total[key] / n_test_batches

        all_test_labels = all_test_labels.repeat(n_traj_samples, 1, 1)
        idx_not_nan = ~torch.isnan(all_test_labels)
        classif_predictions = classif_predictions[idx_not_nan]
        all_test_labels = all_test_labels[idx_not_nan]
        total["auroc"] = 0.0
        total["auprc"] = 0.0
        if torch.sum(all_test_labels) != 0.0:
            self.logger.info(
                "Number of labeled examples: {}".format(len(all_test_labels.reshape(-1))))
            self.logger.info(
                "Number of positive examples: {}".format(
                    int(torch.sum(all_test_labels == 1.0))))
            array_truth = all_test_labels.cpu().numpy().reshape(-1)
            array_predict = classif_predictions.cpu().numpy().reshape(-1)
            total["auroc"] = sk.metrics.roc_auc_score(array_truth, array_predict)
            total["auprc"] = sk.metrics.average_precision_score(array_truth, array_predict)
        else:
            self.logger.warning("Couldn't compute AUC -- all examples are from one class")

        return total

    def run(self) -> None:
        early_stop_metric = getattr(self.args, "early_stop_metric", "val_loss")
        early_stop_mode = self._early_stop_mode()
        best_loss = -float("inf") if early_stop_mode == "max" else float("inf")
        waiting = 0
        durations = []
        initial_state = deepcopy(self.model.state_dict())
        best_checkpoints = {
            "total": {
                "mode": "min",
                "metric": "val_loss",
                "best": float("inf"),
                "state": deepcopy(initial_state),
                "epoch": 0,
            },
            "ce": {
                "mode": "min",
                "metric": "ce_loss",
                "best": float("inf"),
                "state": deepcopy(initial_state),
                "epoch": 0,
            },
            "auprc": {
                "mode": "max",
                "metric": "auprc",
                "best": -float("inf"),
                "state": deepcopy(initial_state),
                "epoch": 0,
            },
            "auroc": {
                "mode": "max",
                "metric": "auroc",
                "best": -float("inf"),
                "state": deepcopy(initial_state),
                "epoch": 0,
            },
        }

        total_epochs = max(self.epochs_max - 1, 0)
        ema_decay = float(getattr(self.args, "ema_decay", 0.0))
        ema_state = None
        swa_start = int(getattr(self.args, "swa_start", 0))
        swa_state = None
        swa_updates = 0
        epoch = 0
        for epoch in range(1, self.epochs_max):
            self.model.train()
            start_time = time.time()
            train_loss_sum = 0.0
            train_batches = 0
            train_desc = f"biclass {self.args.data} epoch {epoch:04d}/{total_epochs:04d}"

            with tqdm(
                self.dltrain,
                desc=train_desc,
                unit="batch",
                leave=True,
                dynamic_ncols=True,
                file=sys.stdout,
            ) as progress:
                for batch in progress:
                    self.optim.zero_grad()
                    train_loss = self.training_step(batch)
                    train_loss.backward()
                    if self.args.clip_gradient:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                    self.optim.step()
                    if ema_decay > 0.0:
                        ema_state = self._update_ema_state_dict(
                            ema_state, self.model.state_dict(), ema_decay)

                    train_loss_sum += self._scalar(train_loss)
                    train_batches += 1

            if swa_start > 0 and epoch >= swa_start:
                swa_updates += 1
                swa_state = self._average_state_dict(
                    swa_state, self.model.state_dict(), swa_updates)

            epoch_duration = time.time() - start_time
            durations.append(epoch_duration)
            self.logger.info(f"[epoch={epoch:04d}] epoch_duration={epoch_duration:.5f}")

            self.model.eval()
            val_metrics = self.validation_step(epoch)
            val_loss_value = self._metric_value(val_metrics, "val_loss")
            early_stop_value = self._metric_value(val_metrics, early_stop_metric)

            if self.scheduler and self.scheduler_type == "plateau":
                scheduler_metric = self._metric_value(
                    val_metrics, getattr(self.args, "plateau_monitor", "auprc"))
                self.scheduler.step(scheduler_metric)
            elif self.scheduler:
                self.scheduler.step()

            if self._is_better(early_stop_value, best_loss, early_stop_mode):
                best_loss = early_stop_value
                waiting = 0
            else:
                waiting += 1

            if val_loss_value < best_checkpoints["total"]["best"]:
                best_checkpoints["total"]["best"] = val_loss_value
                best_checkpoints["total"]["state"] = deepcopy(self.model.state_dict())
                best_checkpoints["total"]["epoch"] = epoch

            ce_value = self._metric_value(val_metrics, "ce_loss")
            auprc_value = self._metric_value(val_metrics, "auprc")
            auroc_value = self._metric_value(val_metrics, "auroc")
            if ce_value < best_checkpoints["ce"]["best"]:
                best_checkpoints["ce"]["best"] = ce_value
                best_checkpoints["ce"]["state"] = deepcopy(self.model.state_dict())
                best_checkpoints["ce"]["epoch"] = epoch
            if auprc_value > best_checkpoints["auprc"]["best"]:
                best_checkpoints["auprc"]["best"] = auprc_value
                best_checkpoints["auprc"]["state"] = deepcopy(self.model.state_dict())
                best_checkpoints["auprc"]["epoch"] = epoch
            if auroc_value > best_checkpoints["auroc"]["best"]:
                best_checkpoints["auroc"]["best"] = auroc_value
                best_checkpoints["auroc"]["state"] = deepcopy(self.model.state_dict())
                best_checkpoints["auroc"]["epoch"] = epoch

            avg_train_loss = train_loss_sum / max(train_batches, 1)
            epoch_summary = (
                f"[epoch={epoch:04d}/{total_epochs:04d}] "
                f"train_loss={avg_train_loss:.5f} "
                f"val_loss={val_loss_value:.5f} "
                f"{self._format_metrics(val_metrics, self._summary_metric_keys())} "
                f"early_stop={early_stop_metric} "
                f"best_stop={best_loss:.5f} "
                f"best_ce={best_checkpoints['ce']['best']:.5f} "
                f"best_auprc={best_checkpoints['auprc']['best']:.5f} "
                f"best_auroc={best_checkpoints['auroc']['best']:.5f} "
                f"lr={self.optim.param_groups[0]['lr']:.2e} "
                f"wait={waiting}/{self.patience} "
                f"time={epoch_duration:.2f}s")
            tqdm.write(epoch_summary, file=sys.stdout)
            self.logger.info(epoch_summary)

            if waiting >= self.patience:
                tqdm.write(
                    f"[early-stop] patience reached at epoch {epoch:04d}; "
                    f"metric={early_stop_metric} best={best_loss:.5f}",
                    file=sys.stdout,
                )
                break

        epoch_duration_mean = float(np.mean(durations)) if durations else float("nan")
        self.logger.info(f"epoch_duration_mean={epoch_duration_mean:.5f}")
        if ema_state is not None:
            best_checkpoints["ema"] = {
                "mode": "last",
                "metric": "ema_decay",
                "best": ema_decay,
                "state": ema_state,
                "epoch": epoch,
            }
        if swa_state is not None:
            best_checkpoints["swa"] = {
                "mode": "last",
                "metric": "swa_start",
                "best": float(swa_start),
                "state": swa_state,
                "epoch": epoch,
            }
        for ckpt_name, ckpt_info in best_checkpoints.items():
            self.model.load_state_dict(ckpt_info["state"])
            test_metrics = self.test_step()
            test_loss_value = self._metric_value(test_metrics, "val_loss")
            test_summary = (
                f"[test@best_{ckpt_name}] "
                f"selected_epoch={ckpt_info['epoch']:04d} "
                f"val_{ckpt_info['metric']}={ckpt_info['best']:.5f} "
                f"test_loss={test_loss_value:.5f} "
                f"{self._format_metrics(test_metrics, self._summary_metric_keys())} "
                f"epoch_duration_mean={epoch_duration_mean:.5f}s")
            tqdm.write(test_summary, file=sys.stdout)
            self.logger.info(test_summary)

        self.model.load_state_dict(best_checkpoints["total"]["state"])

    def finish(self):
        record_experiment(self.args, self.model)
        model_dir = self.proj_path / "temp" / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_dir / (self.args.exp_name + ".pt"))
        logging.shutdown()

