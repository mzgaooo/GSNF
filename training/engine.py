import copy
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from model.gsnf_biclass import Classifier
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


def _metrics(labels, scores):
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def _loader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_prepared,
        num_workers=0,
    )


def _collect(model, loader, device):
    model.eval()
    latent = []
    summary = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            result = model.run_validation(batch)
            value = result["z0_features"]
            if value.dim() > 2:
                value = value.mean(dim=0)
            latent.append(value.reshape(value.size(0), -1).cpu().numpy())
            summary.append(batch["clinical_summary"].cpu().numpy())
            labels.append(batch["truth"].reshape(-1).cpu().numpy())
    return {
        "latent": np.concatenate(latent),
        "summary": np.concatenate(summary),
        "labels": np.concatenate(labels).astype(np.int64),
    }


def _validate_splits(train, validation):
    if train.variable_num != validation.variable_num:
        raise ValueError("training and validation variable dimensions differ")
    if train.static_dim != validation.static_dim:
        raise ValueError("training and validation static dimensions differ")
    if train.summary.shape[1] != validation.summary.shape[1]:
        raise ValueError("training and validation summary dimensions differ")


def run(config, data_root, device):
    seed_everything(config.seed)
    config.device = str(device)
    train_set = PreparedSplit(Path(data_root) / "train.npz")
    validation_set = PreparedSplit(Path(data_root) / "validation.npz")
    _validate_splits(train_set, validation_set)
    config.variable_num = train_set.variable_num
    config.latent_dim = train_set.variable_num
    config.graph_num_nodes = train_set.variable_num
    config.static_dim = train_set.static_dim
    train_loader = _loader(train_set, config.batch_size, True)
    train_eval_loader = _loader(train_set, config.batch_size, False)
    validation_loader = _loader(validation_set, config.batch_size, False)
    model = ModelFactory(config).initialize_biclass_model().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.lr_scheduler_step,
        gamma=config.lr_decay,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_value = (-float("inf"), -float("inf"))
    for epoch in range(1, config.epochs + 1):
        model.train()
        for raw_batch in train_loader:
            batch = _move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            result = model.compute_prediction_results(batch, config.k_iwae)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        logits = []
        labels = []
        with torch.no_grad():
            for raw_batch in validation_loader:
                batch = _move_batch(raw_batch, device)
                result = model.run_validation(batch)
                value = result["label_predictions"]
                if value.dim() > 1:
                    value = value.mean(dim=0)
                logits.append(value.reshape(-1).cpu().numpy())
                labels.append(batch["truth"].reshape(-1).cpu().numpy())
        logits = np.concatenate(logits)
        labels = np.concatenate(labels).astype(np.int64)
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        value = _metrics(labels, probability)
        candidate = (value["auprc"], value["auroc"])
        if candidate > best_value:
            best_value = candidate
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        scheduler.step()
    model.load_state_dict(best_state)
    train_values = _collect(model, train_eval_loader, device)
    validation_values = _collect(model, validation_loader, device)
    classifier = Classifier.fit(
        train_values["summary"],
        train_values["latent"],
        train_values["labels"],
        validation_values["summary"],
        validation_values["latent"],
        validation_values["labels"],
        config.seed,
    )
    test_set = PreparedSplit(Path(data_root) / "test.npz")
    _validate_splits(train_set, test_set)
    test_values = _collect(
        model,
        _loader(test_set, config.batch_size, False),
        device,
    )
    scores = classifier.predict(test_values["summary"], test_values["latent"])
    return {
        "seed": int(config.seed),
        "best_epoch": int(best_epoch),
        "test": _metrics(test_values["labels"], scores),
    }
