import time
import torch

from model.gsnf import GSNF
from model.components import BinaryClassifier
from experiments.utils_metrics import compute_binary_CE_loss



class GSNF_BiClass(GSNF):
    def __init__(
            self,
            args,
            embedding_nn,
            embedding_nn_gnn,
            reconst_mapper,
            flow_solver):

        super().__init__(
            args,
            embedding_nn,
            embedding_nn_gnn,
            reconst_mapper,
            flow_solver)

        self.args = args
        self.classifier = BinaryClassifier(self._classifier_dim(), self.args)

    def _classifier_dim(self):
        pool = getattr(self.args, "classifier_pool", "none")
        dim = self.args.latent_dim
        if pool in ["mean", "last"]:
            dim += self.args.latent_dim
        elif pool == "mean_last":
            dim += 2 * self.args.latent_dim
        if getattr(self.args, "use_static_input", False):
            dim += self._static_dim()
        return dim

    def _static_dim(self):
        static_dim = int(getattr(self.args, "static_dim", 0))
        if static_dim > 0:
            return static_dim
        if getattr(self.args, "static_source", "auto") in ["auto", "summary"]:
            return int(getattr(self.args, "static_summary_vars", 5))
        return 0

    def init_classifier_bias(self, pos_rate):
        if pos_rate is None or pos_rate <= 0.0 or pos_rate >= 1.0:
            return
        last_linear = next(
            (module for module in reversed(self.classifier.layers) if isinstance(module, torch.nn.Linear)),
            None)
        if last_linear is not None and last_linear.bias is not None:
            last_linear.bias.data.fill_(torch.logit(torch.tensor(pos_rate)).item())

    def _last_observed_z(self, sol_z, batch):
        lengths = batch.get("lengths")
        if lengths is None:
            lengths = batch["mask_in"].sum(-1).gt(0).sum(-1)
        last_index = (lengths.long() - 1).clamp_min(0)
        gather_index = last_index.view(1, -1, 1, 1).expand(
            sol_z.size(0), -1, 1, sol_z.size(-1))
        return sol_z.gather(2, gather_index).squeeze(2)

    def _mean_observed_z(self, sol_z, batch):
        observed = batch["mask_in"].sum(-1).gt(0).to(sol_z)
        observed = observed.unsqueeze(0).unsqueeze(-1)
        denom = observed.sum(2).clamp_min(1.0)
        return (sol_z * observed).sum(2) / denom

    def _append_pool_features(self, x_input, forward_info, batch):
        pool = getattr(self.args, "classifier_pool", "none")
        if pool == "none":
            return x_input
        sol_z = forward_info.get("sol_z")
        if sol_z is None:
            raise RuntimeError("classifier_pool requires sol_z, but sol_z is None")
        features = [x_input]
        if pool in ["mean", "mean_last"]:
            features.append(self._mean_observed_z(sol_z, batch))
        if pool in ["last", "mean_last"]:
            features.append(self._last_observed_z(sol_z, batch))
        return torch.cat(features, dim=-1)

    def _summary_static_features(self, batch):
        summary_dim = int(getattr(self.args, "static_summary_vars", 5))
        if summary_dim <= 0:
            return None
        values = batch["data_in"][..., :summary_dim]
        masks = batch["mask_in"][..., :summary_dim]
        weighted = values * masks
        denom = masks.sum(dim=1).clamp_min(1.0)
        return weighted.sum(dim=1) / denom

    def _static_features(self, batch):
        source = getattr(self.args, "static_source", "auto")
        static = None
        if source in ["auto", "batch"]:
            static = batch.get("static_in")
        if static is None and source in ["auto", "summary"]:
            static = self._summary_static_features(batch)
        if static is None:
            return None
        static_dim = self._static_dim()
        if static_dim <= 0:
            return None
        if static.size(-1) > static_dim:
            static = static[..., :static_dim]
        elif static.size(-1) < static_dim:
            pad = static.new_zeros(*static.shape[:-1], static_dim - static.size(-1))
            static = torch.cat([static, pad], dim=-1)
        return static

    def _append_static_features(self, x_input, batch):
        if not getattr(self.args, "use_static_input", False):
            return x_input
        static = self._static_features(batch)
        if static is None:
            raise RuntimeError(
                "use_static_input=True but no static features were found. "
                "Use --static-source summary or provide Static_* columns.")
        while static.dim() < x_input.dim():
            static = static.unsqueeze(0)
        static = static.expand(*x_input.shape[:-1], static.size(-1))
        return torch.cat([x_input, static.to(x_input)], dim=-1)

    def compute_prediction_results(self, batch, k_iwae=1):

        if self.args.train_w_reconstr:
            results, forward_info = self.forward(batch, k_iwae)
        else:
            results, forward_info = self.forward_classification(batch, k_iwae)



        if self.args.classifier_input == "z0":
            x_input = forward_info['initial_state'][:, :, 0, :]
        elif self.args.classifier_input == "zn":
            x_input = forward_info['sol_z'][:, :, -1, :]
        else:
            raise NotImplementedError(f"Unknown classifier_input: {self.args.classifier_input}")
        x_input = self._append_pool_features(x_input, forward_info, batch)
        x_input = self._append_static_features(x_input, batch)
        label_pred = self.classifier(x_input)
        results['forward_time'] = time.time() - self.time_start

        # Compute CE loss
        ce_loss = compute_binary_CE_loss(
            label_pred,
            batch['truth'],
            pos_weight=getattr(self.args, "ce_pos_weight_runtime", 0.0),
            label_smoothing=getattr(self.args, "ce_label_smoothing", 0.0),
            focal_gamma=getattr(self.args, "focal_gamma", 0.0))
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["label_predictions"] = label_pred.detach()

        loss = results['loss']
        if self.args.train_w_reconstr:
            loss = loss + ce_loss * self.args.ratio_ce
        else:
            loss = loss + ce_loss
        results["loss"] = torch.mean(loss)
        results["val_loss"] = results["loss"].detach()

        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch, k_iwae=self.args.k_iwae)

