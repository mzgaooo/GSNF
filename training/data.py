from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _features(values, masks, times, static):
    mask = masks > 0
    observed = np.where(mask, values, 0.0)
    count = mask.sum(axis=0).astype(np.float64)
    denominator = np.maximum(count, 1.0)
    mean = observed.sum(axis=0) / denominator
    centered = np.where(mask, values - mean[None, :], 0.0)
    sample_denominator = np.maximum(count - 1.0, 1.0)
    std = np.sqrt((centered ** 2).sum(axis=0) / sample_denominator)
    std[count < 2] = np.nan
    minimum = np.where(mask, values, np.inf).min(axis=0)
    maximum = np.where(mask, values, -np.inf).max(axis=0)
    first_indices = mask.argmax(axis=0)
    last_indices = mask.shape[0] - 1 - mask[::-1].argmax(axis=0)
    variable_indices = np.arange(mask.shape[1])
    first = values[first_indices, variable_indices].astype(np.float64)
    last = values[last_indices, variable_indices].astype(np.float64)
    first_time = times[first_indices].astype(np.float64)
    last_time = times[last_indices].astype(np.float64)
    absent = count == 0
    for value in (minimum, maximum, first, last, first_time, last_time):
        value[absent] = np.nan
    mean_time = (times[:, None] * mask).sum(axis=0) / denominator
    centered_time = times[:, None] - mean_time[None, :]
    slope_denominator = (centered_time ** 2 * mask).sum(axis=0)
    slope = (
        centered_time * (values - mean[None, :]) * mask
    ).sum(axis=0) / np.where(slope_denominator > 1e-8, slope_denominator, 1.0)
    slope[slope_denominator <= 1e-8] = np.nan
    last_observation = float(times[-1])
    per_variable = np.stack([
        mean,
        std,
        minimum,
        maximum,
        first,
        last,
        np.log1p(count),
        count / max(len(times), 1),
        slope,
        first_time,
        last_observation - last_time,
    ], axis=-1).reshape(-1)
    global_values = np.asarray([
        np.log1p(len(times)),
        last_observation,
        np.log1p(mask.sum()),
        mask.mean(),
    ], dtype=np.float64)
    if static.size >= 9:
        static = static[[0, 2, 3, 8, 4, 5, 6, 7]]
    return np.concatenate([per_variable, global_values, static]).astype(np.float32)


class PreparedSplit(Dataset):
    def __init__(self, path):
        self.path = Path(path)
        arrays = np.load(self.path, allow_pickle=False)
        required = {"times", "values", "mask", "lengths", "labels"}
        missing = sorted(required.difference(arrays.files))
        if missing:
            raise ValueError(f"{self.path} is missing arrays: {', '.join(missing)}")
        self.times = np.asarray(arrays["times"], dtype=np.float32)
        self.values = np.asarray(arrays["values"], dtype=np.float32)
        self.mask = np.asarray(arrays["mask"], dtype=np.float32)
        self.lengths = np.asarray(arrays["lengths"], dtype=np.int64)
        self.labels = np.asarray(arrays["labels"], dtype=np.float32).reshape(-1)
        self.static = np.asarray(
            arrays["static"] if "static" in arrays.files else
            np.empty((len(self.labels), 0)),
            dtype=np.float32,
        )
        self.ids = np.asarray(
            arrays["ids"] if "ids" in arrays.files else np.arange(len(self.labels)),
            dtype=np.int64,
        )
        if self.values.ndim != 3 or self.values.shape != self.mask.shape:
            raise ValueError("values and mask must have shape [N,T,V]")
        if self.times.shape != self.values.shape[:2]:
            raise ValueError("times must have shape [N,T]")
        if self.lengths.shape != (len(self.labels),):
            raise ValueError("lengths must have shape [N]")
        if self.static.ndim != 2 or self.static.shape[0] != len(self.labels):
            raise ValueError("static must have shape [N,S]")
        if "summary" in arrays.files:
            self.summary = np.asarray(arrays["summary"], dtype=np.float32)
        else:
            raw_values = np.asarray(
                arrays["raw_values"] if "raw_values" in arrays.files else self.values,
                dtype=np.float64,
            )
            raw_times = np.asarray(
                arrays["raw_times"] if "raw_times" in arrays.files else self.times,
                dtype=np.float64,
            )
            self.summary = np.stack([
                _features(
                    raw_values[index, :int(self.lengths[index])],
                    self.mask[index, :int(self.lengths[index])],
                    raw_times[index, :int(self.lengths[index])],
                    self.static[index].astype(np.float64),
                )
                for index in range(len(self.labels))
            ])
        if self.summary.ndim != 2 or self.summary.shape[0] != len(self.labels):
            raise ValueError("summary must have shape [N,F]")

    @property
    def variable_num(self):
        return int(self.values.shape[-1])

    @property
    def static_dim(self):
        return int(self.static.shape[-1])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        length = int(self.lengths[index])
        if length < 1 or length > self.values.shape[1]:
            raise ValueError(f"invalid sequence length at row {index}: {length}")
        return {
            "idx": int(self.ids[index]),
            "times": self.times[index, :length],
            "values": self.values[index, :length],
            "mask": self.mask[index, :length],
            "label": self.labels[index],
            "static": self.static[index],
            "summary": self.summary[index],
        }


def _delta_features(times, masks):
    batch, length, variables = masks.shape
    delta = torch.zeros_like(masks)
    last_seen = torch.zeros(batch, variables, device=masks.device, dtype=masks.dtype)
    has_seen = torch.zeros(batch, variables, device=masks.device, dtype=torch.bool)
    previous_time = torch.zeros(batch, device=masks.device, dtype=masks.dtype)
    for step in range(length):
        current_time = times[:, step]
        observed = masks[:, step].gt(0)
        valid = observed.any(dim=-1)
        elapsed = (current_time - previous_time).clamp_min(0).unsqueeze(-1)
        current = torch.where(has_seen, last_seen + elapsed, torch.zeros_like(last_seen))
        current = current * valid.unsqueeze(-1).to(masks)
        delta[:, step] = current
        last_seen = torch.where(observed, torch.zeros_like(last_seen), current)
        has_seen = has_seen | observed
        previous_time = torch.where(valid, current_time, previous_time)
    return delta / delta.amax(dim=1, keepdim=True).clamp_min(1.0)


def collate_prepared(items):
    batch_size = len(items)
    length = max(item["times"].shape[0] for item in items)
    variables = items[0]["values"].shape[-1]
    times = torch.zeros(batch_size, length, dtype=torch.float32)
    values = torch.zeros(batch_size, length, variables, dtype=torch.float32)
    masks = torch.zeros_like(values)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    for row, item in enumerate(items):
        current = item["times"].shape[0]
        times[row, :current] = torch.from_numpy(item["times"])
        values[row, :current] = torch.from_numpy(item["values"])
        masks[row, :current] = torch.from_numpy(item["mask"])
        lengths[row] = current
    return {
        "idx": torch.tensor([item["idx"] for item in items], dtype=torch.long),
        "times_in": times,
        "times_out": times.clone(),
        "data_in": values,
        "data_out": values.clone(),
        "mask_in": masks,
        "mask_out": masks.clone(),
        "delta_in": _delta_features(times, masks),
        "lengths": lengths,
        "truth": torch.tensor(
            [item["label"] for item in items], dtype=torch.float32).unsqueeze(-1),
        "static_in": torch.from_numpy(np.stack([item["static"] for item in items])),
        "clinical_summary": torch.from_numpy(
            np.stack([item["summary"] for item in items])),
    }
