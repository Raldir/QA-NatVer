import argparse
import json
import os
import re
from collections import Counter

from tqdm import tqdm

from env import ABSOLUTE_PATH
from src.data.data_reader import DatasetReader


class DanFeverReader(DatasetReader):
    def __init__(self, split, is_few_shot, granularity=None):
        DatasetReader.__init__(self, split, is_few_shot, granularity)
        self.dataset = "danfever"
        # Set static to fever since no training data exists right now
        if split == "train":
            raise Exception("No training set defined for {}".format(self.dataset))
        elif split == "validation":
            self.claim_file = os.path.join(ABSOLUTE_PATH, "data", self.dataset, "da_fever.tsv")
            self.proofver_file = os.path.join(
                ABSOLUTE_PATH, "data", "fever", "proofver_data", "val_with_ids.target"
            )  # Set static to fever since no other have proof files for now
            self.retrieved_evidence_file = os.path.join(
                ABSOLUTE_PATH, "data", self.dataset, "retrieved_evidence", "bm25_validation.txt"
            )

        self.corpus_path = os.path.join(ABSOLUTE_PATH, "data", self.dataset, "da_wikipedia.tsv")

    def read_annotations(self):
        with open(self.claim_file, "r", encoding="utf-8") as f_in:
            # open qrels file if provided
            for i, line in enumerate(f_in):
                if i == 0:
                    continue
                content = line.strip().replace('"', "").split("\t")
                _, qid, query, _, label, _, evidence = content
                if label == "Supported":
                    label = "SUPPORTS"
                elif label == "Refuted":
                    label = "REFUTES"
                elif label == "NotEnoughInfo":
                    label = "NOT ENOUGH INFO"
                yield (qid, query, label, evidence)

    def read_corpus(self):
        documents = {}
        sentences = {}
        sentences_buffer = {}
        skip_next = False
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip() == "" or i < 2:
                    continue
                if skip_next:
                    skip_next = False
                    continue
                if len(line.split("\t")) > 1:
                    doc_title = line.split("\t")[1].replace('"', "")
                    skip_next = True
                    count = 0
                    for sid, sentence in sentences_buffer.items():
                        sentences[doc_title + "_" + str(count)] = sentence
                        count += 1
                    documents[doc_title] = " SEP ".join([y for _, y in sentences_buffer.items()])
                    sentences_buffer = {}
                    continue
                content = line.strip()
                content = re.sub(r"<a.*?>|</a>", "", content)  # Remove <a> tags
                sentences_buffer[str(i)] = content

            if self.granularity == "sentence":
                # each li in "lines" is of the format: (sentence id)\t(sentence)[\t(tag)\t...\t(tag)]
                docs = []
                docs_titles = []
                # count=0
                for i, li in sentences.items():
                    docs.append(li)
                    docs_titles.append(i)
                    # count+=
            elif self.granularity == "pipeline":
                docs = []
                docs_titles = []
                for i, li in documents.items():
                    docs.append(li)
                    docs_titles.append(i)
            else:  # args.granularity == 'paragraph'
                docs = []
                docs_titles = []
                for i, li in documents.items():
                    docs.append(li.replace(" SEP ", " "))
                    docs_titles.append(i)

            for i, doc in enumerate(docs):
                yield {"id": docs_titles[i], "contents": doc}
