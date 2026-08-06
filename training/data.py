from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PreparedSplit(Dataset):
    """Read one already-prepared split without dataset-specific processing.

    Each split is an NPZ file with padded arrays:
    ``times [N,T]``, ``values [N,T,37]``, ``mask [N,T,37]``,
    ``lengths [N]``, ``labels [N]``, ``static [N,9]``, and
    ``summary [N,419]``. Optional ``ids [N]`` defaults to row indices.
    """

    def __init__(self, path):
        self.path = Path(path)
        arrays = np.load(self.path, allow_pickle=False)
        required = {"times", "values", "mask", "lengths", "labels", "static", "summary"}
        missing = sorted(required.difference(arrays.files))
        if missing:
            raise ValueError(f"{self.path} is missing arrays: {', '.join(missing)}")
        self.times = np.asarray(arrays["times"], dtype=np.float32)
        self.values = np.asarray(arrays["values"], dtype=np.float32)
        self.mask = np.asarray(arrays["mask"], dtype=np.float32)
        self.lengths = np.asarray(arrays["lengths"], dtype=np.int64)
        self.labels = np.asarray(arrays["labels"], dtype=np.float32).reshape(-1)
        self.static = np.asarray(arrays["static"], dtype=np.float32)
        self.summary = np.asarray(arrays["summary"], dtype=np.float32)
        self.ids = np.asarray(
            arrays["ids"] if "ids" in arrays.files else np.arange(len(self.labels)),
            dtype=np.int64,
        )
        if self.values.ndim != 3 or self.values.shape != self.mask.shape:
            raise ValueError("values and mask must both have shape [N,T,V]")
        if self.times.shape != self.values.shape[:2]:
            raise ValueError("times must have shape [N,T]")
        if self.values.shape[-1] != 37:
            raise ValueError("the focused release expects 37 dynamic variables")
        if self.static.shape != (len(self.labels), 9):
            raise ValueError("static must have shape [N,9]")
        if self.summary.shape != (len(self.labels), 419):
            raise ValueError("summary must have shape [N,419]")

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
