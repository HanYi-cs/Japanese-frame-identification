# -*-coding:utf-8-*-

import argparse
import glob
import os

import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fn_path", type=str, default="../data/framenet/verb")
    parser.add_argument("--setting_path", type=str, default="../data/framenet/setting")

    parser.add_argument(
        "--input_c4_path", type=str, default="../data/c4/verb/train_00000"
    )

    parser.add_argument("--output_path", type=str, default="../data/c4/setting")

    parser.add_argument("--min_ex", type=int, default=5)
    return parser.parse_args()


def make_dir_path(args):
    path_dict = {
        "input_c4": args.input_c4_path,
        "input_fn": args.input_fn_path,
        "setting": args.setting_path,
        "output": args.output_path,
    }
    for key, path in path_dict.items():
        path_dict[key] = path + "/"
        if "output" in key:
            if not os.path.isdir(path_dict[key]):
                os.makedirs(path_dict[key])
    return path_dict


def main():
    args = parse_args()
    path_dict = make_dir_path(args)

    df1 = pd.read_json(
        path_dict["input_fn"] + "exemplars.jsonl", orient="records", lines=True
    )
    df2 = pd.read_json(
        path_dict["setting"] + "supervised.jsonl", orient="records", lines=True
    )
    df_fn = pd.merge(df1, df2, on="ex_idx")

    sets_dict = {}
    for column in [c for c in list(df2.columns) if c != "ex_idx"]:
        sets_dict[column + "-testset"] = {}
        sets_dict[column + "-framenet"] = {}
        sets_dict[column + "-c4"] = {}
        for verb in sorted(list(set(df_fn[df_fn[column] == "test"]["verb"]))):
            sets_dict[column + "-testset"][verb] = "test"
        for verb in sorted(set(df_fn["verb"])):
            sets_dict[column + "-framenet"][verb] = "use"

    df_c4_list = []
    for path in tqdm(sorted(glob.glob(path_dict["input_c4"] + "*_0000.jsonl"))):
        df_verb = pd.read_json(path, orient="records", lines=True)
        df_verb["verb"] = df_verb["verb"].fillna("nan")
        if len(df_verb) <= args.min_ex:
            continue

        df_test = df_verb.copy()
        for column, verb2sets in sets_dict.items():
            if "-testset" in column:
                df_test[column] = df_test["verb"].map(verb2sets)
                df_test[column] = df_test[column].fillna("disuse")
            elif "-framenet" in column:
                df_test[column] = df_test["verb"].map(verb2sets)
                df_test[column] = df_test[column].map({"use": "test"})
                df_test[column] = df_test[column].fillna("disuse")
            elif "-c4" in column:
                df_test.loc[:, column] = "test"
        df_c4_list.append(df_test)
    df_c4 = pd.concat(df_c4_list, axis=0)

    df_c4 = (
        df_c4.reset_index(drop=True).reset_index().rename(columns={"index": "ex_idx"})
    )

    df_output = df_c4[["ex_idx"] + list(sets_dict.keys())]
    df_output.to_json(
        path_dict["output"] + "supervised.jsonl",
        orient="records",
        force_ascii=False,
        lines=True,
    )

    df_output = df_c4[[c for c in df_c4.columns if c not in list(sets_dict.keys())]]
    df_output.to_json(
        path_dict["output"] + "exemplars.jsonl",
        orient="records",
        force_ascii=False,
        lines=True,
    )


if __name__ == "__main__":
    main()
