from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from experiments.data_mimic4 import DatasetBiClass


DATASET_CONFIG = {
    "p12": {
        "folder": "P12data",
        "time_divisor": 2880.0,
    },
    "p19": {
        "folder": "P19data",
        "time_divisor": 60.0,
    },
}


def _select_indices(indices, num_samples, seed):
    if num_samples is None or num_samples <= 0 or num_samples >= len(indices):
        return indices
    rng = np.random.RandomState(seed)
    selected = rng.choice(indices, size=num_samples, replace=False)
    return np.asarray(selected, dtype=indices.dtype)


def _sample_records(pt_list, labels, idx_train, idx_val, idx_test, args):
    num_samples = getattr(args, "num_samples", -1)
    if num_samples is None or num_samples <= 0:
        return idx_train, idx_val, idx_test

    train_n = int(round(num_samples * 0.8))
    valid_n = int(round(num_samples * 0.1))
    test_n = max(1, num_samples - train_n - valid_n)
    train_n = max(1, train_n)
    valid_n = max(1, valid_n)

    idx_train = _select_indices(idx_train, train_n, args.random_state)
    idx_val = _select_indices(idx_val, valid_n, args.random_state + 13)
    idx_test = _select_indices(idx_test, test_n, args.random_state + 29)
    return idx_train, idx_val, idx_test


def _seed_split(labels, seed):
    indices = np.arange(len(labels), dtype=np.int64)
    idx_train, idx_vt = train_test_split(
        indices,
        train_size=0.8,
        random_state=int(seed),
        shuffle=True)
    idx_val, idx_test = train_test_split(
        idx_vt,
        train_size=0.5,
        random_state=int(seed),
        shuffle=True)
    return (
        np.asarray(idx_train, dtype=np.int64),
        np.asarray(idx_val, dtype=np.int64),
        np.asarray(idx_test, dtype=np.int64),
    )


def _load_arrays(base_path):
    processed_path = base_path / "processed_data"
    pt_path = processed_path / "PTdict_list.npy"
    label_path = processed_path / "arr_outcomes.npy"
    if not pt_path.exists():
        raise FileNotFoundError(f"Missing processed records: {pt_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing outcome labels: {label_path}")
    pt_list = np.load(pt_path, allow_pickle=True)
    labels = np.load(label_path, allow_pickle=True)[:, -1].astype(np.float32)
    return pt_list, labels


def _get_train_stats(pt_list, idx_train):
    train_records = pt_list[idx_train]
    _, feature_dim = train_records[0]["arr"].shape
    sums = np.zeros(feature_dim, dtype=np.float64)
    sums_sq = np.zeros(feature_dim, dtype=np.float64)
    counts = np.zeros(feature_dim, dtype=np.float64)

    for record in train_records:
        arr = np.asarray(record["arr"], dtype=np.float32)
        length = int(record.get("length", arr.shape[0]))
        arr = arr[:length]
        mask = arr > 0
        sums += (arr * mask).sum(axis=0)
        sums_sq += ((arr ** 2) * mask).sum(axis=0)
        counts += mask.sum(axis=0)

    mean = np.zeros(feature_dim, dtype=np.float32)
    std = np.ones(feature_dim, dtype=np.float32)
    observed = counts > 0
    mean[observed] = (sums[observed] / counts[observed]).astype(np.float32)
    var = (sums_sq[observed] / counts[observed]) - mean[observed] ** 2
    std[observed] = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
    return mean, std


def _get_static_stats(pt_list, idx_train):
    static_rows = []
    for raw_idx in idx_train:
        record = pt_list[int(raw_idx)]
        if "extended_static" in record:
            static_rows.append(np.asarray(record["extended_static"], dtype=np.float32))
    if not static_rows:
        return None, None
    static = np.stack(static_rows, axis=0)
    mean = np.nanmean(static, axis=0).astype(np.float32)
    std = np.nanstd(static, axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def _records_to_frame(pt_list, labels, indices, mean, std, time_divisor, time_max,
                      static_mean=None, static_std=None, use_static=False):
    rows = []
    max_time = float(time_max)
    for raw_idx in indices:
        record = pt_list[int(raw_idx)]
        arr = np.asarray(record["arr"], dtype=np.float32)
        times = np.asarray(record["time"], dtype=np.float32).reshape(-1)
        length = int(record.get("length", arr.shape[0]))
        arr = arr[:length]
        times = times[:length]

        mask = arr > 0
        value = ((arr - mean) / (std + 1e-18)) * mask
        scaled_times = times / time_divisor
        static_row = None
        if use_static and "extended_static" in record:
            static_row = np.asarray(record["extended_static"], dtype=np.float32)
            if static_mean is not None and static_std is not None:
                static_row = (static_row - static_mean) / (static_std + 1e-6)
            static_row = np.nan_to_num(static_row, nan=0.0, posinf=0.0, neginf=0.0)

        for t, value_row, mask_row in zip(scaled_times, value, mask):
            if t <= 0:
                t = 1e-4
            if max_time > 0 and t > max_time:
                continue
            if not mask_row.any():
                continue
            row = {"ID": int(raw_idx), "Time": float(t)}
            for i, v in enumerate(value_row):
                row[f"Value_{i}"] = float(v)
            for i, m in enumerate(mask_row):
                row[f"Mask_{i}"] = float(m)
            if static_row is not None:
                for i, v in enumerate(static_row):
                    row[f"Static_{i}"] = float(v)
            rows.append(row)

    data_df = pd.DataFrame(rows)
    label_df = pd.DataFrame({
        "ID": np.asarray(indices, dtype=np.int64),
        "labels": labels[np.asarray(indices, dtype=np.int64)].astype(np.float32),
    })
    return data_df, label_df


def get_physionet_tvt_datasets(args, data_dir, logger):
    data_name = args.data.lower()
    if data_name not in DATASET_CONFIG:
        raise ValueError(f"Unsupported PhysioNet dataset: {args.data}")

    config = DATASET_CONFIG[data_name]
    base_path = Path(data_dir) / "IrregularTimeSeriesDatasets" / config["folder"]
    pt_list, labels = _load_arrays(base_path)
    idx_train, idx_val, idx_test = _seed_split(labels, getattr(args, "random_state", 1))
    idx_train, idx_val, idx_test = _sample_records(
        pt_list, labels, idx_train, idx_val, idx_test, args)

    mean, std = _get_train_stats(pt_list, idx_train)
    static_mean, static_std = _get_static_stats(pt_list, idx_train)
    use_static = bool(getattr(args, "use_static_input", False))
    if use_static and static_mean is not None:
        args.static_dim = max(int(getattr(args, "static_dim", 0)), int(static_mean.shape[0]))
    time_max = args.time_max / config["time_divisor"] if args.time_max > 1 else args.time_max
    train_df, train_labels = _records_to_frame(
        pt_list, labels, idx_train, mean, std, config["time_divisor"], time_max,
        static_mean, static_std, use_static)
    val_df, val_labels = _records_to_frame(
        pt_list, labels, idx_val, mean, std, config["time_divisor"], time_max,
        static_mean, static_std, use_static)
    test_df, test_labels = _records_to_frame(
        pt_list, labels, idx_test, mean, std, config["time_divisor"], time_max,
        static_mean, static_std, use_static)

    label_df = pd.concat([train_labels, val_labels, test_labels], ignore_index=True)
    logger.info(f"{args.data}: seed_split random_state={args.random_state}")
    logger.info(f"Number of samples for training: {train_df['ID'].nunique()}")
    logger.info(f"Number of samples for validation: {val_df['ID'].nunique()}")
    logger.info(f"Number of samples for testing: {test_df['ID'].nunique()}")
    logger.info(f"Number of positives in training: {int(train_labels['labels'].sum())}")
    logger.info(f"Number of positives in validation: {int(val_labels['labels'].sum())}")
    logger.info(f"Number of positives in testing: {int(test_labels['labels'].sum())}")

    train = DatasetBiClass(train_df, label_df=label_df, ts_full=args.ts_full)
    val = DatasetBiClass(val_df, label_df=label_df, ts_full=args.ts_full)
    test = DatasetBiClass(test_df, label_df=label_df, ts_full=args.ts_full)
    return train, val, test
