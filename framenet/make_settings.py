# -*-coding:utf-8-*-

import argparse
import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    LANGUAGE = ["en", "ja"][0]
    SETTING = ["all", "zero", "vf01", "vf02", "vf05", "vf10", "vf20"][0]

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="../data/framenet/stanza")
    parser.add_argument("--output_path", type=str, default="../data/framenet/setting")

    parser.add_argument("--language", type=str, default=LANGUAGE)
    parser.add_argument("--setting", type=str, default=SETTING)
    parser.add_argument("--n_splits", type=int, default=3)
    return parser.parse_args()


def make_dir_path(args):
    path_dict = {
        "input": "/".join([args.input_path, args.language]),
        "output": "/".join([args.output_path, args.language]),
    }
    for key, path in path_dict.items():
        path_dict[key] = path + "/"
        if "output" in key:
            if not os.path.isdir(path_dict[key]):
                os.makedirs(path_dict[key])
    return path_dict


def make_n_split(v_list, n_splits):
    return [
        [v_list[i] for i in indices]
        for indices in np.array_split(range(len(v_list)), n_splits)
    ]


def use_idx(df, n_splits):
    vf2idx_list = {}
    for df_dict in df.to_dict("records"):
        vf, ex_idx = df_dict["verb_frame"], df_dict["ex_idx"]
        if vf not in vf2idx_list:
            vf2idx_list[vf] = [ex_idx]
        else:
            vf2idx_list[vf].append(ex_idx)

    all_idx_list = []
    for vf, idx_list in vf2idx_list.items():
        random.seed(0)
        all_idx_list += random.sample(idx_list, min(n_splits, len(idx_list)))
    return all_idx_list


def main():
    args = parse_args()
    path_dict = make_dir_path(args)

    df = pd.read_json(
        path_dict["input"] + "exemplars.jsonl", orient="records", lines=True
    )

    if not os.path.isfile(path_dict["output"] + "exemplars.jsonl"):
        df.to_json(
            path_dict["output"] + "exemplars.jsonl",
            orient="records",
            force_ascii=False,
            lines=True,
        )

    all_list, v2_list = [], []
    for verb, _ in df.groupby(["verb", "verb_frame"]).count().index:
        if verb not in all_list:
            all_list.append(verb)
        else:
            if verb not in v2_list:
                v2_list.append(verb)
    v1_list = sorted(set(all_list) - set(v2_list))
    v2_list = sorted(v2_list)

    random.seed(0)
    random.shuffle(v1_list)
    random.shuffle(v2_list)

    n_v1_list = make_n_split(v1_list, args.n_splits)
    n_v2_list = make_n_split(v2_list, args.n_splits)[::-1]
    n_v_list = [v1 + v2 for v1, v2 in zip(n_v1_list, n_v2_list)] * 2

    setting_i_list = []
    for i in tqdm(range(args.n_splits)):
        test_v_list = n_v_list[i]
        dev_v_list = n_v_list[i + 1]
        train_v_list = sum(n_v_list[i + 2 : i + args.n_splits], [])

        df_test = df[df["verb"].isin(test_v_list)]
        df_dev = df[df["verb"].isin(dev_v_list)]
        df_train = df[df["verb"].isin(train_v_list)]

        if "all" in args.setting:
            setting_i = "_".join([args.setting, str(args.n_splits), str(i)])
            setting_i_list.append(setting_i)
            df[setting_i] = "disuse"
            for tdt, df_item in zip(
                ["train", "dev", "test"], [df_train, df_dev, df_test]
            ):
                df.loc[list(df_item.index), setting_i] = tdt
        elif "zero" in args.setting:
            setting_i = "_".join([args.setting, str(args.n_splits), str(i)])
            setting_i_list.append(setting_i)
            df[setting_i] = "disuse"
            for tdt, df_item in zip(
                ["train", "dev", "test"],
                [
                    df_train[~df_train["frame_name"].isin(list(df_test["frame_name"]))],
                    df_dev,
                    df_test,
                ],
            ):
                df.loc[list(df_item.index), setting_i] = tdt
        elif "vf" in args.setting:
            n = int(args.setting[-2:])
            setting_i = "_".join([args.setting, str(args.n_splits), str(i)])
            setting_i_list.append(setting_i)
            df[setting_i] = "disuse"
            for tdt, df_item in zip(
                ["train", "dev", "test"],
                [
                    df_train[df_train["ex_idx"].isin(use_idx(df_train, n))],
                    df_dev,
                    df_test,
                ],
            ):
                df.loc[list(df_item.index), setting_i] = tdt

    df_output = df[["ex_idx"] + setting_i_list]
    df_output.to_json(
        path_dict["output"] + "_".join([args.setting, str(args.n_splits)]) + ".jsonl",
        orient="records",
        force_ascii=False,
        lines=True,
    )


if __name__ == "__main__":
    main()
