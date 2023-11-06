import json
import os

from src.data.data_reader import DatasetReader
from src.utils.util import ROOT_DIR


class FeverReader(DatasetReader):
    def __init__(self, split, is_few_shot, granularity=None):
        DatasetReader.__init__(self, split, is_few_shot, granularity)
        self.dataset = "fever"
        if split == "train" and not is_few_shot:
            self.claim_file = os.path.join(ROOT_DIR, "data", "fever", "train.jsonl")
            self.retrieved_evidence_file = os.path.join(
                ROOT_DIR, "data", "fever", "retrieved_evidence", "stammbach_evidence_train.txt"
            )
            self.proofver_file = os.path.join(ROOT_DIR, "data", "fever", "proofver_data", "train_with_ids.target")
        elif split == "train":
            self.claim_file = os.path.join(ROOT_DIR, "data", "fever", "train.jsonl")
            self.retrieved_evidence_file = os.path.join(
                ROOT_DIR, "data", "fever", "retrieved_evidence", "stammbach_evidence_train.txt"
            )
            self.proofver_file = os.path.join(
                ROOT_DIR, "data", "fever", "proofver_data", "train_with_ids_fewshot.target"
            )
        elif split == "validation":
            self.claim_file = os.path.join(ROOT_DIR, "data", self.dataset, "shared_task_dev.jsonl")
            self.retrieved_evidence_file = os.path.join(
                ROOT_DIR, "data", self.dataset, "retrieved_evidence", "stammbach_evidence_validation.txt"
            )
            self.proofver_file = os.path.join(
                ROOT_DIR, "data", "fever", "proofver_data", "val_with_ids.target"
            )  # Set static to fever since no other have proof files for now

        self.granularity = granularity

    def read_annotations(self):
        with open(self.claim_file, "r", encoding="utf-8") as f_in:
            # open qrels file if provided

            for line in f_in:
                line_json = json.loads(line.strip())
                qid = line_json["id"]
                query = line_json["claim"]
                if "label" in line_json:  # no "label" field in test datasets
                    label = line_json["label"]
                    if label == "NOT ENOUGH INFO":
                        evidences = [[]]
                    else:
                        #     continue
                        evidences = []
                        for annotator in line_json["evidence"]:
                            ev = []
                            for evidence in annotator:
                                ev.append("{}_{}".format(evidence[2], evidence[3]))
                            evidences.append(ev)
                else:  # Test mode, no labels
                    label = None
                    evidences = None
                yield (qid, query, label, evidences)

    def read_corpus(self, input_path):
        pass
