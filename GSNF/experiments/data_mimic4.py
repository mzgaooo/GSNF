import re

import numpy as np
import pandas as pd
import sklearn.model_selection
import torch
from torch.utils.data import Dataset

import utils


def merge_timestamps(df_data, down_times):
    df_data["Time"] = df_data["Time"].div(down_times).apply(np.floor) * down_times
    df_data.reset_index(inplace=True)
    df_data = df_data.groupby(["ID", "Time"], as_index=False).max()
    df_data.set_index("ID", inplace=True)
    return df_data


def mask_random(df_in, ratio):
    df_out = df_in.copy()
    mask = np.random.choice([0, 1], size=df_in.shape, p=[ratio, 1 - ratio])
    return df_out * mask


def filter_tvt(df_all, logger, args):
    ids_before = df_all.loc[df_all["Time"] < args.next_start].index.unique()
    ids_after = df_all.loc[df_all["Time"] > args.next_start].index.unique()
    ids_selected = set(ids_before) & set(ids_after)
    df_all = df_all.loc[list(ids_selected)]
    logger.info("Number of samples: {}".format(len(ids_selected)))

    if args.time_max < df_all["Time"].max():
        df_all = df_all.loc[df_all["Time"] <= args.time_max]

    if args.mask_drop_rate > 0:
        value_cols = [col.startswith("Value") for col in df_all.columns]
        mask_cols = [col.startswith("Mask") for col in df_all.columns]
        value = df_all.loc[:, value_cols].values
        mask = df_all.loc[:, mask_cols].values
        mask_new = mask_random(mask, args.mask_drop_rate)
        df_all.loc[:, value_cols] = value * mask_new
        df_all.loc[:, mask_cols] = mask_new
        df_all = df_all.loc[df_all.loc[:, mask_cols].sum(axis=1) > 0]

    ids_train, ids_vt = sklearn.model_selection.train_test_split(
        df_all.index.unique(),
        train_size=0.8,
        random_state=args.random_state,
        shuffle=True,
    )
    ids_valid, ids_test = sklearn.model_selection.train_test_split(
        ids_vt,
        train_size=0.5,
        random_state=args.random_state,
        shuffle=True,
    )
    return df_all.loc[ids_train], df_all.loc[ids_valid], df_all.loc[ids_test]


def load_tvt(args, m4_path, logger):
    if args.data != "MIMIC4":
        raise ValueError(f"Unsupported MIMIC4 dataset: {args.data}")

    data_mimic4 = pd.read_csv(m4_path / "mimic4_full_dataset.csv", index_col=0)
    if args.num_samples != -1:
        tvt_ids = pd.DataFrame(data_mimic4.index.unique(), columns=["ID"]).sample(
            n=args.num_samples, random_state=args.random_state)
        data_tvt = data_mimic4.loc[tvt_ids["ID"]]
    else:
        data_tvt = data_mimic4

    return filter_tvt(data_tvt, logger, args)


def scale_tvt(args, data_train, data_validation, data_test, logger):
    logger.info("Number of samples for training: {}".format(data_train.index.nunique()))
    logger.info("Number of samples for validation: {}".format(data_validation.index.nunique()))
    logger.info("Number of samples for testing: {}".format(data_test.index.nunique()))
    time_max = args.time_max
    if args.down_times > 1:
        data_train = merge_timestamps(data_train, args.down_times)
        data_validation = merge_timestamps(data_validation, args.down_times)
        data_test = merge_timestamps(data_test, args.down_times)

    if args.t_offset > 0:
        data_train.loc[:, "Time"] = data_train["Time"] + args.t_offset
        data_validation.loc[:, "Time"] = data_validation["Time"] + args.t_offset
        data_test.loc[:, "Time"] = data_test["Time"] + args.t_offset
        time_max += args.t_offset

    if args.time_scale == "time_max":
        data_train.loc[:, "Time"] = data_train["Time"] / time_max
        data_validation.loc[:, "Time"] = data_validation["Time"] / time_max
        data_test.loc[:, "Time"] = data_test["Time"] / time_max
    elif args.time_scale in ["self_max", "max"]:
        data_train.loc[:, "Time"] = data_train["Time"] / data_train["Time"].max()
        data_validation.loc[:, "Time"] = data_validation["Time"] / data_validation["Time"].max()
        data_test.loc[:, "Time"] = data_test["Time"] / data_test["Time"].max()
    elif args.time_scale == "constant":
        data_train.loc[:, "Time"] = data_train["Time"] / args.time_constant
        data_validation.loc[:, "Time"] = data_validation["Time"] / args.time_constant
        data_test.loc[:, "Time"] = data_test["Time"] / args.time_constant

    value_cols = [c.startswith("Value") for c in data_train.columns]
    value_cols = data_train.iloc[:, value_cols]
    mask_cols = [("Mask" + x[5:]) for x in value_cols]

    data_train.dropna(inplace=True)
    for item in zip(value_cols, mask_cols):
        val_train = data_train.loc[data_train[item[1]].astype("bool"), item[0]]
        val_validation = data_validation.loc[data_validation[item[1]].astype("bool"), item[0]]
        val_test = data_test.loc[data_test[item[1]].astype("bool"), item[0]]

        if getattr(args, "scale_train_only", False):
            mean_tv, std_tv = utils.calc_mean_std(val_train)
            mean_t, std_t = mean_tv, std_tv
        else:
            df_tv = pd.concat([val_train, val_validation])
            df_tvt = pd.concat([val_train, val_validation, val_test])
            mean_tv, std_tv = utils.calc_mean_std(df_tv)
            mean_t, std_t = utils.calc_mean_std(df_tvt)

        data_train.loc[data_train[item[1]].astype("bool"), item[0]] = (
            val_train - mean_tv) / std_tv
        data_validation.loc[data_validation[item[1]].astype("bool"), item[0]] = (
            val_validation - mean_tv) / std_tv
        data_test.loc[data_test[item[1]].astype("bool"), item[0]] = (
            val_test - mean_t) / std_t

    data_train.dropna(inplace=True)
    data_validation.dropna(inplace=True)
    data_test.dropna(inplace=True)
    return data_train, data_validation, data_test


def get_mimic4_tvt_datasets(args, data_dir, logger):
    m4_path = data_dir / "mimic4" / "processed"
    data_train, data_validation, data_test = load_tvt(args, m4_path, logger)
    data_train, data_validation, data_test = scale_tvt(
        args, data_train, data_validation, data_test, logger)

    label_data = pd.read_csv(m4_path / "mortality_labels.csv")
    label_data["labels"] = label_data["labels"].astype(float)
    train = DatasetBiClass(data_train.reset_index(), label_df=label_data, ts_full=args.ts_full)
    val = DatasetBiClass(data_validation.reset_index(), label_df=label_data, ts_full=args.ts_full)
    test = DatasetBiClass(data_test.reset_index(), label_df=label_data, ts_full=args.ts_full)
    return train, val, test


class DatasetBiClass(Dataset):
    def __init__(self, in_df, label_df, ts_full=False):
        self.in_df = in_df
        adm_ids = self.in_df.loc[:, "ID"].unique()
        self.label_df = label_df[label_df["ID"].isin(adm_ids)].copy()
        self.length = len(adm_ids)
        self.variable_num = sum(col.startswith("Value") for col in self.in_df.columns)

        map_dict = dict(zip(adm_ids, np.arange(self.length)))
        self.in_df.loc[:, "ID"] = self.in_df.loc[:, "ID"].map(map_dict)
        self.label_df.loc[:, "ID"] = self.label_df["ID"].map(map_dict)

        self.in_df = self.in_df.astype(np.float32)
        self.in_df.ID = self.in_df.ID.astype(int)
        self.label_df["labels"] = self.label_df["labels"].astype(np.float32)
        if ts_full:
            self.in_df.Time = self.in_df.Time.astype(int)
        self.in_df.set_index("ID", inplace=True)
        self.in_df.sort_values("Time", inplace=True)
        self.label_df.set_index("ID", inplace=True)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        samples = self.in_df.loc[[idx]]
        label = self.label_df.loc[idx].values
        item = {"idx": idx, "truth": label, "samples": samples}
        static_cols = [col for col in samples.columns if col.startswith("Static_")]
        if static_cols:
            item["static"] = samples.loc[:, static_cols].iloc[0].astype(np.float32).values
        return item


def _compute_delta_features(times, masks):
    if times.dim() != 2 or masks.dim() != 3:
        raise ValueError("delta features expect times [B,T] and masks [B,T,D]")
    batch_size, length, num_vars = masks.shape
    delta = torch.zeros_like(masks)
    last_seen = times.new_zeros(batch_size, num_vars)
    has_seen = torch.zeros(batch_size, num_vars, dtype=torch.bool, device=times.device)
    prev_time = times.new_zeros(batch_size)

    for step in range(length):
        current_time = times[:, step]
        observed_step = masks[:, step, :].gt(0)
        valid_step = masks[:, step, :].sum(dim=-1).gt(0)
        elapsed = (current_time - prev_time).clamp_min(0.0).unsqueeze(-1)
        current_delta = torch.where(has_seen, last_seen + elapsed, torch.zeros_like(last_seen))
        current_delta = current_delta * valid_step.unsqueeze(-1).to(masks)
        delta[:, step, :] = current_delta
        last_seen = torch.where(observed_step, torch.zeros_like(last_seen), current_delta)
        has_seen = has_seen | observed_step
        prev_time = torch.where(valid_step, current_time, prev_time)

    max_delta = delta.amax(dim=1, keepdim=True).clamp_min(1.0)
    return delta / max_delta


def collate_fn_biclass(batch, num_vars, args):
    device = args.device if hasattr(args, "device") else torch.device("cpu")
    value_cols = [col.startswith("Value") for col in batch[0]["samples"].columns]
    mask_cols = [col.startswith("Mask") for col in batch[0]["samples"].columns]

    if args.ts_full:
        data_list = []
        truth_list = []
        for item in batch:
            df_new = pd.DataFrame(
                0.0,
                index=np.arange(args.num_times),
                columns=item["samples"].columns[1:],
            )
            df_tmp = item["samples"].set_index("Time")
            df_new.loc[df_tmp.index] = df_tmp
            value = df_new.loc[:, value_cols[1:]].values
            mask = df_new.loc[:, mask_cols[1:]].values
            if args.mask_type == "cumsum":
                mask = mask.cumsum(axis=0) * mask
            data_list.append(np.concatenate((value, mask), axis=-1))
            truth_list.append(item["truth"])

        data_batch = torch.tensor(np.array(data_list, dtype=np.float32)).to(device).permute(0, 2, 1)
        combined_truth = torch.tensor(np.array(truth_list)).to(device)
        return {"data": data_batch, "truth": combined_truth, "mask": mask}

    values_list = []
    masks_list = []
    times_list = []
    len_list = []
    truth_list = []
    static_list = []
    for item in batch:
        values_list.append(item["samples"].loc[:, value_cols].values)
        masks_list.append(item["samples"].loc[:, mask_cols].values)
        ts = item["samples"]["Time"].values
        times_list.append(ts)
        len_list.append(len(ts))
        truth_list.append(item["truth"])
        if "static" in item:
            static_list.append(np.asarray(item["static"], dtype=np.float32))

    max_len = max(len_list)
    combined_values = torch.from_numpy(np.stack([
        np.concatenate([values, np.zeros((max_len - len_t, num_vars), dtype=np.float32)], 0)
        for values, len_t in zip(values_list, len_list)
    ], 0)).to(device)
    combined_masks = torch.from_numpy(np.stack([
        np.concatenate([mask, np.zeros((max_len - len_t, num_vars), dtype=np.float32)], 0)
        for mask, len_t in zip(masks_list, len_list)
    ], 0)).to(device)
    combined_times = torch.from_numpy(np.stack([
        np.concatenate([times, np.zeros(max_len - len_t, dtype=np.float32)], 0)
        for times, len_t in zip(times_list, len_list)
    ], 0).astype(np.float32)).to(device)
    combined_truth = torch.tensor(np.array(truth_list)).to(device)
    lengths = torch.tensor(len_list).to(device)

    assert combined_times[:, 0].gt(0).all()

    if args.first_dim == "time_series":
        combined_values = combined_values.permute(1, 0, 2)
        combined_masks = combined_masks.permute(1, 0, 2)
        combined_times = combined_times.permute(1, 0)

    data_dict = {
        "times_in": combined_times,
        "data_in": combined_values,
        "mask_in": combined_masks,
        "truth": combined_truth,
        "lengths": lengths,
    }
    if len(static_list) == len(batch):
        data_dict["static_in"] = torch.from_numpy(
            np.stack(static_list, 0).astype(np.float32)).to(device)
    if getattr(args, "use_delta_input", False):
        data_dict["delta_in"] = _compute_delta_features(combined_times, combined_masks)
    return data_dict
