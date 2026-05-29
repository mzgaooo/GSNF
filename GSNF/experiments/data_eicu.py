import pandas as pd

from experiments.data_mimic4 import DatasetBiClass, filter_tvt, scale_tvt


def load_tvt(args, path_eicu, logger):
    path_processed = path_eicu / "processed"
    data_eicu = pd.read_csv(path_processed / "eicu_data.csv", index_col=0)

    if args.num_samples != -1:
        tvt_ids = pd.DataFrame(data_eicu.index.unique(), columns=["ID"]).sample(
            n=args.num_samples, random_state=args.random_state)
        data_tvt = data_eicu.loc[tvt_ids["ID"]]
    else:
        data_tvt = data_eicu

    return filter_tvt(data_tvt, logger, args)


def get_eicu_tvt_datasets(args, data_dir, logger):
    path_eicu = data_dir / "eicu"
    data_train, data_validation, data_test = load_tvt(args, path_eicu, logger)
    data_train, data_validation, data_test = scale_tvt(
        args, data_train, data_validation, data_test, logger)

    label_data = pd.read_csv(path_eicu / "processed" / "eicu_labels.csv")
    label_data["labels"] = label_data["labels"].astype(float)
    train = DatasetBiClass(data_train.reset_index(), label_df=label_data, ts_full=args.ts_full)
    val = DatasetBiClass(data_validation.reset_index(), label_df=label_data, ts_full=args.ts_full)
    test = DatasetBiClass(data_test.reset_index(), label_df=label_data, ts_full=args.ts_full)
    return train, val, test
