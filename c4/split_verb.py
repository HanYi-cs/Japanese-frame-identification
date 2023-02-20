# -*-coding:utf-8-*-

import argparse
import os
import re

import pandas as pd
from tqdm import tqdm


def parse_args():
    LANGUAGE = ["en", "ja"][1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="../data/c4/stanza")
    parser.add_argument("--output_path", type=str, default="../data/c4/verb")
    parser.add_argument("--language", type=str, default=LANGUAGE)
    parser.add_argument("--tv", type=str, default="train")
    parser.add_argument("--file_id", type=int, default=0)
    parser.add_argument("--part_id_start", type=int, default=0)  # en:0 ja:0
    parser.add_argument("--part_id_end", type=int, default=356)  # en:356 ja:86
    parser.add_argument("--n_files", type=int, default=5)
    parser.add_argument("--n_sents", type=int, default=100)
    return parser.parse_args()


def make_dir_path(args):
    path_dict = {
        "input": os.path.join(
            args.input_path,
            args.language,
            "_".join([args.tv, str(args.file_id).zfill(5)]),
        ),
        "output": os.path.join(
            args.output_path,
            args.language,
            "_".join([args.tv, str(args.file_id).zfill(5)]),
        ),
    }
    for key, path in path_dict.items():
        path_dict[key] = path + "/"
        if "output" in key:
            if not os.path.isdir(path_dict[key]):
                os.makedirs(path_dict[key])
    return path_dict


if __name__ == "__main__":

    args = parse_args()
    path_dict = make_dir_path(args)

    if args.language == "en":
        pattern = "[a-z][a-z-]*"
    elif args.language == "ja":
        pattern = "[ぁ-んァ-ン一-龥][ぁ-んァ-ン一-龥ー]*"

    output_dict, verb2count = {}, {}
    finished_verb_list = []
    for part_id in tqdm(range(args.part_id_start, args.part_id_end + 1)):
        df = pd.read_json(
            path_dict["input"] + "exemplars_" + str(part_id).zfill(4) + ".jsonl",
            orient="records",
            lines=True,
        )

        for verb in tqdm(sorted(set(df["verb"]))):
            if not re.fullmatch(pattern, verb):
                continue

            if verb in finished_verb_list:
                continue

            if verb not in output_dict:
                output_dict[verb] = df[df["verb"] == verb]
            else:
                output_dict[verb] = pd.concat(
                    [output_dict[verb], df[df["verb"] == verb]], axis=0
                )
            if verb not in verb2count:
                verb2count[verb] = 0
            for i in range(len(output_dict[verb]) // args.n_sents):
                df_verb = output_dict[verb][: args.n_sents]

                df_verb.to_json(
                    path_dict["output"]
                    + "_".join([verb, str(verb2count[verb]).zfill(4)])
                    + ".jsonl",
                    orient="records",
                    force_ascii=False,
                    lines=True,
                )
                output_dict[verb] = output_dict[verb][args.n_sents :]
                verb2count[verb] += 1

                if verb2count[verb] > args.n_files - 1:
                    finished_verb_list.append(verb)
                    break

    for verb, df_verb in output_dict.items():
        if verb in finished_verb_list:
            continue

        if len(df_verb) == 0:
            continue

        df_verb.to_json(
            path_dict["output"]
            + "_".join([verb, str(verb2count[verb]).zfill(4)])
            + ".jsonl",
            orient="records",
            force_ascii=False,
            lines=True,
        )
