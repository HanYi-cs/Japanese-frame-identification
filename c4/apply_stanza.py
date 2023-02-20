# -*-coding:utf-8-*-
import argparse
import os
import re
import unicodedata

import pandas as pd
import stanza
from datasets import load_dataset
from tqdm import tqdm


def parse_args():
    LANGUAGE = ["en", "ja"][1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="../data/c4/stanza")
    parser.add_argument("--language", type=str, default=LANGUAGE)
    parser.add_argument("--tv", type=str, default="train")
    parser.add_argument("--file_id", type=int, default=0)
    parser.add_argument("--part_id_start", type=int, default=40)  # en:0 ja:0
    parser.add_argument("--part_id_end", type=int, default=59)  # en:356 ja:86
    return parser.parse_args()


def make_dir_path(args):
    path_dict = {
        "output": os.path.join(
            args.output_path,
            args.language,
            "_".join([args.tv, str(args.file_id).zfill(5)]),
        )
    }
    for key, path in path_dict.items():
        path_dict[key] = path + "/"
        if "output" in key:
            if not os.path.isdir(path_dict[key]):
                os.makedirs(path_dict[key])
    return path_dict


def clean_text(text):
    text = re.sub("\s", " ", text)
    text = "".join(
        [
            char
            for char in text
            if unicodedata.category(char)
            not in ["Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp", "So"]
        ]
    )
    return text


def make_children_dict(doc):
    children_dict = {}
    for sent_id, sent in enumerate(doc.sentences):
        children_dict[sent_id] = {}
        for word in sent.words:
            if word.head not in children_dict[sent_id]:
                children_dict[sent_id][word.head] = [word.id]
            else:
                children_dict[sent_id][word.head].append(word.id)
    return children_dict


def make_word_list(doc, children_dict):
    word_list, count, word_count = [], 0, 0
    for sent_id, sent in enumerate(doc.sentences):
        c_dict = children_dict[sent_id]
        for word in sent.words:
            word_dict = word.to_dict()
            word_dict.update(
                {"id": count, "sent_id": sent_id, "word_id": int(word.id) - 1}
            )
            if word.head != 0:
                word_dict.update(
                    {
                        "head": word.head - 1 + word_count,
                        "head_text": sent.words[word.head - 1].text,
                    }
                )
            else:
                word_dict.update({"head": -1, "head_text": "[ROOT]"})
            if word.id in c_dict:
                word_dict.update(
                    {
                        "children": [i - 1 + word_count for i in c_dict[word.id]],
                        "children_text": [i - 1 + word_count for i in c_dict[word.id]],
                        "n_lefts": len(
                            [i for i in c_dict[word.id] if i < int(word.id)]
                        ),
                        "n_rights": len(
                            [i for i in c_dict[word.id] if i > int(word.id)]
                        ),
                    }
                )
            else:
                word_dict.update(
                    {"children": [], "children_text": [], "n_lefts": 0, "n_rights": 0}
                )
            word_list.append(word_dict)
            count += 1
        word_count += len(sent.words)
    return word_list


if __name__ == "__main__":

    args = parse_args()
    path_dict = make_dir_path(args)

    nlp = stanza.Pipeline(args.language)

    if args.language == "en":
        data_files = (
            "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-"
            + ".".join([args.tv, str(args.file_id).zfill(5) + "-of-01024.json.gz"])
        )
    elif args.language == "ja":
        data_files = (
            "https://huggingface.co/datasets/allenai/c4/resolve/main/multilingual/c4-ja.tfrecord-"
            + str(args.file_id).zfill(5)
            + "-of-01024.json.gz",
        )

    dataset = load_dataset("json", data_files=data_files)
    text_list = dataset[args.tv]["text"]

    for part_id in tqdm(range(args.part_id_start, args.part_id_end + 1)):
        output_list, output_word_list, count = [], [], 0
        for doc_id in tqdm(range(part_id * 1000, (part_id + 1) * 1000)):
            if doc_id >= len(text_list):
                continue
            doc = nlp(clean_text(text_list[doc_id]))
            children_dict = make_children_dict(doc)
            word_list = make_word_list(doc, children_dict)

            df_wl = pd.DataFrame(word_list)
            for sent_id in sorted(set(df_wl["sent_id"])):
                df_sent = df_wl[df_wl["sent_id"] == sent_id]
                for word_dict in df_sent.to_dict("records"):
                    if word_dict["upos"] == "VERB":
                        output_list.append(
                            {
                                "tv": args.tv,
                                "file_id": args.file_id,
                                "part_id": part_id,
                                "id": count,
                                "doc_id": doc_id,
                                "sent_id": sent_id,
                                "word_id": word_dict["word_id"],
                                "verb": word_dict["lemma"],
                                "word": word_dict["text"],
                                "text_widx": " ".join(df_sent["text"]),
                            }
                        )
                        output_word_list.append(
                            {"id": count, "word_list": list(df_sent.to_dict("records"))}
                        )
                        count += 1

        if len(output_list) != 0:
            df_output = pd.DataFrame(output_list)
            df_output.to_json(
                path_dict["output"] + "exemplars_" + str(part_id).zfill(4) + ".jsonl",
                orient="records",
                force_ascii=False,
                lines=True,
            )
            df_owl = pd.DataFrame(output_word_list)
            df_owl.to_json(
                path_dict["output"] + "word_list_" + str(part_id).zfill(4) + ".jsonl",
                orient="records",
                force_ascii=False,
                lines=True,
            )
