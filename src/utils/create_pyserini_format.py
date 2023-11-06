"""
Author: Rami Aly
"""

import argparse
import json
import os

from src.utils.util import ROOT_DIR
from src.data.danfever_reader import DanFeverReader
from src.data.fever_reader import FeverReader
"""
The name of this file is a bit misleading since the original FEVER dataset is
also in JSONL format. This script converts them into a JSONL format compatible
with anserini.
"""


def convert_collection(args):
    print("Converting collection...")
    if args.dataset == "fever":
        reader = FeverReader("validation", True, args.granularity)
    if args.dataset == "danfever":
        reader = DanFeverReader("validation", True, args.granularity)

    iterator = reader.read_corpus()
    doc_index = 0
    file_index = 0
    for i, output_dict in enumerate(iterator):
        if doc_index % args.max_docs_per_file == 0:
            if doc_index > 0:
                output_jsonl_file.close()
            output_path = os.path.join(args.output_folder, f"docs{file_index:02d}.json")
            output_jsonl_file = open(output_path, "w", encoding="utf-8", newline="\n")
            file_index += 1
        output_jsonl_file.write(json.dumps(output_dict) + "\n")
        doc_index += 1

        if doc_index % 100000 == 0:
            print("Converted {} docs in {} files".format(doc_index, file_index))

    output_jsonl_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts FEVER jsonl wikipedia dump to anserini jsonl files.")
    parser.add_argument(
        "--max_docs_per_file", default=1000000, type=int, help="Maximum number of documents in each jsonl file."
    )
    parser.add_argument(
        "--granularity",
        required=True,
        choices=["paragraph", "sentence", "pipeline"],
        help='The granularity of the source documents to index. Either "paragraph" or "sentence".',
    )
    parser.add_argument("--dataset", required=True, help="The dataset.")
    args = parser.parse_args()

    output_folder = os.path.join(ROOT_DIR, "data", args.dataset, "corpus_pyserini_" + str(args.granularity))

    args.output_folder = output_folder

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    convert_collection(args)

    print("Done!")
