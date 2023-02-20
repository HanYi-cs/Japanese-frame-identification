# -*-coding:utf-8-*-

import argparse
import glob
import os

import pandas as pd
import xmltodict
from tqdm import tqdm


def parse_args():
    LANGUAGE = ["en", "ja"][0]

    if LANGUAGE == "en":
        input_path = "../data/source/framenet_v17/lu"
    elif LANGUAGE == "ja":
        input_path = "../data/source/XML_Data20220314/lu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=input_path)
    parser.add_argument("--output_path", type=str, default="../data/framenet/raw")
    parser.add_argument("--language", type=str, default=LANGUAGE)
    return parser.parse_args()


def make_dir_path(args):
    path_dict = {
        "input": args.input_path,
        "output": "/".join([args.output_path, args.language]),
    }
    for key, path in path_dict.items():
        path_dict[key] = path + "/"
        if "output" in key:
            if not os.path.isdir(path_dict[key]):
                os.makedirs(path_dict[key])
    return path_dict


def make_exemplar_dataframe(file_list):
    ex_list = []
    for file in tqdm(file_list):
        with open(file, "r") as f:
            doc = xmltodict.parse(f.read())
        if "subCorpus" not in doc["lexUnit"]:
            continue
        subcorpus_dict_list = (
            doc["lexUnit"]["subCorpus"]
            if type(doc["lexUnit"]["subCorpus"]) is list
            else [doc["lexUnit"]["subCorpus"]]
        )
        for subcorpus_dict in subcorpus_dict_list:
            if "sentence" not in subcorpus_dict:
                continue
            sentence_dict_list = (
                subcorpus_dict["sentence"]
                if type(subcorpus_dict["sentence"]) is list
                else [subcorpus_dict["sentence"]]
            )
            for sentence_dict in sentence_dict_list:
                for anno_dict in sentence_dict["annotationSet"]:
                    if anno_dict["@status"] != "MANUAL":
                        continue
                    target, fe = [], [[], {}]
                    for layer_dict in anno_dict["layer"]:
                        if "label" not in layer_dict:
                            continue
                        label_dict_list = (
                            layer_dict["label"]
                            if type(layer_dict["label"]) is list
                            else [layer_dict["label"]]
                        )
                        for label_dict in label_dict_list:
                            if "@start" in label_dict:
                                start = int(label_dict["@start"])
                                end = int(label_dict["@end"])
                                if layer_dict["@name"] == "Target":
                                    target.append([start, end])
                                elif layer_dict["@name"] == "FE":
                                    fe[0].append([start, end, label_dict["@name"]])
                            else:
                                fe[1][label_dict["@name"]] = label_dict["@itype"]
                    if len(target) != 0:
                        ex_dict = {
                            "frame_name": doc["lexUnit"]["@frame"],
                            "frame_id": doc["lexUnit"]["@frameID"],
                            "lu_name": doc["lexUnit"]["@name"],
                            "lu_id": doc["lexUnit"]["@ID"],
                            "text": sentence_dict["text"],
                            "ex_id": sentence_dict["@ID"],
                            "target": target,
                            "fe": fe,
                        }
                        ex_list.append(ex_dict)
    return pd.DataFrame(ex_list)


def main():
    args = parse_args()
    path_dict = make_dir_path(args)

    df = make_exemplar_dataframe(glob.glob(path_dict["input"] + "lu*.xml"))
    df.to_json(
        path_dict["output"] + "exemplars.jsonl",
        orient="records",
        force_ascii=False,
        lines=True,
    )


if __name__ == "__main__":
    main()
