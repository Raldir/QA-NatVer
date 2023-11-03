import json
import os
from collections import Counter
from env import ABSOLUTE_PATH

from src.data.data_reader import DatasetReader

class SymmetricFeverReader(DatasetReader):

    def __init__(self, split, is_few_shot, granularity=None):
        DatasetReader.__init__(self, split, is_few_shot, granularity)
        self.dataset = "fever_symmetric"
            # Set static to fever since no training data exists right now
        if split == "train":
            raise Exception("No training set defined for {}".format(self.dataset))
        elif split == "validation":
            self.claim_file = os.path.join(ABSOLUTE_PATH, "data", self.dataset, "symmetric_dev_v2.jsonl")
            self.proofver_file = os.path.join(ABSOLUTE_PATH, "data", "fever", "proofver_data", "val_with_ids.target") # Set static to fever since no other have proof files for now


    def read_annotations(self):
        with open(self.claim_file, "r", encoding="utf-8") as f_in:
            # open qrels file if provided

            for line in f_in:
                line_json = json.loads(line.strip())
                qid = line_json["id"]
                query = line_json["claim"]
                label = line_json["label"]
                evidences = line_json["evidence"]
                evidences = self.process_sentence(evidences)
                yield (qid, query, label, evidences)
    
    def read_corpus(self, input_path):
        pass                