import copy
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from model.gsnf_biclass import ContextAdapter
from model.model_factory import ModelFactory
from training.data import PreparedSplit, collate_prepared


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _binary_metrics(labels, logits):
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def _collect_predictions(model, loader, device, k_iwae):
    model.eval()
    logits, z0, summaries, labels = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            result = model.run_validation(batch)
            pred = result["label_predictions"]
            latent = result["z0_features"]
            if pred.dim() > 1:
                pred = pred.mean(dim=0)
            if latent.dim() > 2:
                latent = latent.mean(dim=0)
            logits.append(pred.reshape(-1).cpu().numpy())
            z0.append(latent.reshape(latent.size(0), -1).cpu().numpy())
            summaries.append(batch["clinical_summary"].cpu().numpy())
            labels.append(batch["truth"].reshape(-1).cpu().numpy())
    return {
        "logits": np.concatenate(logits),
        "z0": np.concatenate(z0),
        "summary": np.concatenate(summaries),
        "labels": np.concatenate(labels).astype(np.int64),
    }


def run(config, data_root, device):
    seed_everything(config.seed)
    config.device = str(device)
    train_set = PreparedSplit(Path(data_root) / "train.npz")
    validation_set = PreparedSplit(Path(data_root) / "validation.npz")
    test_set = PreparedSplit(Path(data_root) / "test.npz")
    loaders = {
        "train": DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_prepared,
            num_workers=0,
        ),
        "validation": DataLoader(
            validation_set,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_prepared,
            num_workers=0,
        ),
        "test": DataLoader(
            test_set,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_prepared,
            num_workers=0,
        ),
    }

    model = ModelFactory(config).initialize_biclass_model().to(device)
    positive_rate = float(np.mean(train_set.labels))
    model.init_classifier_bias(positive_rate)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_value = (-float("inf"), -float("inf"))
    for epoch in range(1, config.epochs + 1):
        model.train()
        for raw_batch in loaders["train"]:
            batch = _move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            result = model.compute_prediction_results(batch, config.k_iwae)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        validation = _collect_predictions(model, loaders["validation"], device, config.k_iwae)
        validation_metrics = _binary_metrics(
            validation["labels"], validation["logits"])
        candidate_value = (
            validation_metrics["auprc"], validation_metrics["auroc"])
        if candidate_value > best_value:
            best_value = candidate_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    train_pred = _collect_predictions(model, loaders["train"], device, config.k_iwae)
    validation_pred = _collect_predictions(
        model, loaders["validation"], device, config.k_iwae)
    adapter = ContextAdapter.fit(
        train_pred["summary"], train_pred["logits"], train_pred["labels"],
        validation_pred["summary"], validation_pred["logits"], validation_pred["labels"],
        config.seed,
    )
    test_pred = _collect_predictions(model, loaders["test"], device, config.k_iwae)
    final_score = adapter.predict(test_pred["summary"], test_pred["logits"])
    test_metrics = {
        "auroc": float(roc_auc_score(test_pred["labels"], final_score)),
        "auprc": float(average_precision_score(test_pred["labels"], final_score)),
    }

    result = {
        "seed": int(config.seed),
        "best_epoch": best_epoch,
        "selection_metric": "validation_auprc",
        "test": test_metrics,
    }
    return result
