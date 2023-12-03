import copy
import json
import os
import random
import sys
import traceback

from tqdm import tqdm

from src.data.chunking_and_alignment import DynamicSentenceAligner
from src.data.danfever_reader import DanFeverReader
from src.data.fever_reader import FeverReader
from src.utils.util import ROOT_DIR


class DatasetProcessor(object):
    def __init__(
        self,
        dataset,
        split,
        num_samples=32,
        seed=42,
        stratified_sampling=False,
        is_few_shot=False,
        dynamic_parsing=False,
        ev_sentence_concat_op=" </s> ",
        use_retrieved_evidence="True",
        num_retrieved_evidence=3,
        overwrite_data=False,
        is_debug=False,
        alignment_model="bert",
        matching_method="inter",
        sentence_transformer="sentence-transformers/all-mpnet-base-v2",
        max_chunks=6,
        alignment_mode="simalign",
        loose_matching=True,
    ):
        assert use_retrieved_evidence in [
            "True",
            "False",
            "Only",
            True,
            False,
        ], "Value for use_retrieved_evidence not known, select from {}".format(["True", "False", "Only"])

        self.dataset = dataset
        self.num_samples = num_samples
        self.stratified_sampling = stratified_sampling
        self.split = split
        self.seed = seed
        self.is_few_shot = is_few_shot
        self.dynamic_parsing = dynamic_parsing
        self.split_name = split + "-fewshot" if self.is_few_shot and "train" in split else split
        self.num_retrieved_evidence = num_retrieved_evidence
        self.use_retrieved_evidence = use_retrieved_evidence
        self.ev_sentence_concat_op = ev_sentence_concat_op
        self.alignment_mode = alignment_mode
        self.max_chunks = max_chunks
        self.alignment_model = alignment_model
        self.matching_method = matching_method
        self.loose_matching = loose_matching
        self.sentence_transformer = sentence_transformer

        self.claims = {}
        self.labels = {}
        self.sentence_evidence = {}
        self.claims_parsed = {}
        self.claims_parsed_hierarchy = {}
        self.alignments = {}
        self.proofver_proofs = {}

        if split != "test":  # Load existing data for training and dev if it exists
            self._set_save_path()

            print("Trying to find data from {} ...".format(self.save_path))
            if os.path.exists(self.save_path) and not overwrite_data:
                print("Load existing data from {} ...".format(self.save_path))
                self.load_existing_data(is_debug)

                label_stats = {"SUPPORTS": 0, "REFUTES": 0, "NOT ENOUGH INFO": 0}
                all_keys = set([])
                for key, lab in self.labels.items():
                    acc_key = str(key).split("0000000000")[0]
                    if acc_key in all_keys:
                        continue
                    all_keys.add(acc_key)
                    label_stats[lab] += 1
                print("Label Distributions: ", label_stats)
                return

            self.setup_dataset()

        else:
            test_path = os.path.join(ROOT_DIR, "data", "test.jsonl")
            with open(test_path, "r") as f_in:
                lines = f_in.readlines()
                for line in lines:
                    content = json.loads(line)
                    qid = content["id"]
                    self.claims[qid] = content["claim"]
                    self.sentence_evidence[qid] = self.ev_sentence_concat_op.join(content["evidence"])
                    self.labels[qid] = content["label"] if "label" in content else "SUPPORTS"
            self.save_path = os.path.join(ROOT_DIR, "data", "test_processed.jsonl")

        self._align_samples()

        return

    def _align_samples(self):
        # Use proofver for alignment or our multi-granular alignment method
        if self.alignment_mode == "proofver":
            for qid, claim in tqdm(self.claims.items()):
                # Sometimes proofver fails to parse a claim
                if qid in self.proofver_proofs:
                    proof = self.proofver_proofs[qid]
                else:
                    proof = [("Hello", "Hello", "=")]
                claim_parsed = [x[0] for x in proof]
                claim_parsed_hierarchy = [i for i, x in enumerate(claim_parsed)]
                alignment = [(x[0], x[1]) for x in proof]
                self.claims_parsed[qid] = claim_parsed
                self.claims_parsed_hierarchy[qid] = [claim_parsed_hierarchy]
                self.alignments[qid] = alignment
        else:
            aligner = DynamicSentenceAligner(
                dataset=self.dataset,
                alignment_model=self.alignment_model,
                matching_method=self.matching_method,
                sentence_transformer=self.sentence_transformer,
                num_retrieved_evidence=self.num_retrieved_evidence,
                max_chunks=self.max_chunks,
                alignment_mode=self.alignment_mode,
                loose_matching=self.loose_matching,
                dynamic_parsing=self.dynamic_parsing,
            )

            for qid, claim in tqdm(self.claims.items()):
                # if qid != 134135:
                #     continue
                try:
                    print(qid, claim, self.sentence_evidence[qid])
                    aligned = aligner.align_sample(qid, claim, self.sentence_evidence[qid])
                except:
                    traceback.print_exc()
                    print(
                        "Alignment Error for qid {} and claim {} and evidence {}".format(
                            qid, claim, self.sentence_evidence[qid]
                        )
                    )
                self.claims_parsed[qid] = aligned["claim_parsed"]
                self.claims_parsed_hierarchy[qid] = aligned["claim_parsed_hierarchy"]
                self.alignments[qid] = aligned["alignment"]

        self._save_data()

    def _set_save_path(self):
        dataset_path = self.dataset
        if "train" in self.split:
            samples_text = str(self.num_samples)
            if self.stratified_sampling:
                samples_text += "_stratified"
            print("ALIGNMENT MODE", self.alignment_mode)
            self.save_path = os.path.join(
                ROOT_DIR,
                "data",
                "fever",
                "processed_{}_num_samples_{}_seed_{}_use_retr_{}_retr_evidence_{}_dp_{}_alignment_mode_{}_max_chunks_{}_alignment_model_{}_matching_method_{}_loose_matching_{}.jsonl".format(
                    self.split_name,
                    samples_text,
                    self.seed,
                    self.use_retrieved_evidence,
                    self.num_retrieved_evidence,
                    self.dynamic_parsing,
                    self.alignment_mode,
                    self.max_chunks,
                    self.alignment_model,
                    self.matching_method,
                    self.loose_matching,
                ),
            )
        else:  # Do not add num samples argument since evaluation is on entire data
            self.save_path = os.path.join(
                ROOT_DIR,
                "data",
                dataset_path,
                "processed_{}_use_retr_{}_retr_evidence_{}_dp_{}_alignment_mode_{}_max_chunks_{}_alignment_model_{}_matching_method_{}_loose_matching_{}.jsonl".format(
                    self.split_name,
                    self.use_retrieved_evidence,
                    self.num_retrieved_evidence,
                    self.dynamic_parsing,
                    self.alignment_mode,
                    self.max_chunks,
                    self.alignment_model,
                    self.matching_method,
                    self.loose_matching,
                ),
            )

    def _save_data(self):
        with open(self.save_path, "w") as f_out:
            for key in self.claims:
                dictc = {}
                if key in self.proofver_proofs or self.split == "test":
                    claim = self.claims[key]
                    claims_parsed = self.claims_parsed[key]
                    claim_parsed_hierarchy = self.claims_parsed_hierarchy[key]
                    alignment = self.alignments[key]
                    evidence = self.sentence_evidence[key]
                    proof = self.proofver_proofs[key] if key in self.proofver_proofs else []
                    verdict = self.labels[key]
                    dictc["id"] = key
                    dictc["claim"] = claim
                    dictc["verdict"] = verdict
                    dictc["evidence"] = evidence
                    dictc["proof"] = proof
                    dictc["claim_parsed"] = claims_parsed
                    dictc["claim_parsed_hierarchy"] = claim_parsed_hierarchy
                    dictc["alignment"] = alignment
                    f_out.write("{}\n".format(json.dumps(dictc)))

    def _load_evidence(self, gold_evidences, already_filled=False):
        if self.use_retrieved_evidence in ["False", False]:
            sentence_evidence_ids = gold_evidences
            sentence_evidence_ids_retrieved = self.dataset_reader.get_sentence_predictions(
                num_docs=self.num_retrieved_evidence
            )
            sentence_evidence_ids_new = {}
            for key, label in self.labels.items():
                if label == "NOT ENOUGH INFO":
                    sentence_evidence_ids_new[key] = sentence_evidence_ids_retrieved[key]
                    if already_filled:
                        all_sentences = []
                        for ev_id in sentence_evidence_ids_new[key]:
                            compiled_sentence = self.dataset_reader.get_sentence_content_from_id(
                                ev_id, self.searcher_sentences
                            )
                            if self.dataset in [
                                "fever",
                                "danfever",
                            ]:  # Concatenate title only for datasets with relevant titles
                                compiled_sentence_w_title = (
                                    "[ "
                                    + self.dataset_reader.process_title(" ".join(ev_id.split("_")[:-1]))
                                    + " ] "
                                    + self.dataset_reader.process_sentence(compiled_sentence)
                                )
                            else:
                                title = self.dataset_reader.doc_id_to_title[int(ev_id.split("_")[0])]
                                compiled_sentence_w_title = (
                                    "[ " + title + " ] " + self.dataset_reader.process_sentence(compiled_sentence)
                                )
                            all_sentences.append(compiled_sentence_w_title)
                        sentence_evidence_ids_new[key] = self.dataset_reader.process_sentence(
                            self.ev_sentence_concat_op.join(all_sentences)
                        )
                else:  # TODO: FIX What if already filled sentences contain more than one?
                    sentence_evidence_ids_new[key] = sentence_evidence_ids[key]
            sentence_evidence_ids = copy.deepcopy(sentence_evidence_ids_new)
        elif self.use_retrieved_evidence == "Only":
            print("USING ONLY RETRIEVED EVIDENCE.")
            sentence_evidence_ids = self.dataset_reader.get_sentence_predictions(num_docs=self.num_retrieved_evidence)
        elif self.use_retrieved_evidence in ["True", True]:
            sentence_evidence_ids = self.dataset_reader.get_sentence_predictions(num_docs=self.num_retrieved_evidence)
            if "train" in self.split_name:
                for key, value in sentence_evidence_ids.items():
                    # Do nothing if retrieved evidence already fully covers gold evidence
                    if key not in gold_evidences or all([x in value for x in gold_evidences[key]]):
                        continue
                    else:
                        for ev in gold_evidences[key]:
                            if ev in value:
                                continue
                            else:
                                position = random.randint(1, 100)
                                if position < 60:
                                    sentence_evidence_ids[key].insert(0, ev)
                                elif position < 80:
                                    sentence_evidence_ids[key].insert(1, ev)
                                elif position < 90:
                                    sentence_evidence_ids[key].insert(2, ev)
                                elif position < 95:
                                    sentence_evidence_ids[key].insert(3, ev)
                                elif position < 100:
                                    sentence_evidence_ids[key].insert(4, ev)
                                sentence_evidence_ids[key] = sentence_evidence_ids[key][:5]

        if not already_filled:
            for key, _ in self.claims.items():
                ev_ids = sentence_evidence_ids[key]
                all_sentences = []
                for ev_id in ev_ids:
                    compiled_sentence = self.dataset_reader.get_sentence_content_from_id(
                        ev_id, self.searcher_sentences
                    )
                    if self.dataset in [
                        "fever",
                        "danfever",
                    ]:  # Concatenate title only for datasets with relevant titles
                        compiled_sentence_w_title = (
                            "[ "
                            + self.dataset_reader.process_title(" ".join(ev_id.split("_")[:-1]))
                            + " ] "
                            + self.dataset_reader.process_sentence(compiled_sentence)
                        )
                    else:
                        title = self.dataset_reader.doc_id_to_title[int(ev_id.split("_")[0])]
                        compiled_sentence_w_title = (
                            "[ " + title + " ] " + self.dataset_reader.process_sentence(compiled_sentence)
                        )
                    all_sentences.append(compiled_sentence_w_title)
                if not all_sentences:
                    self.sentence_evidence[key] = ""
                else:
                    self.sentence_evidence[key] = self.dataset_reader.process_sentence(
                        self.ev_sentence_concat_op.join(all_sentences)
                    )
        else:
            for key, _ in self.claims.items():
                self.sentence_evidence[key] = sentence_evidence_ids[key]

    def load_existing_data(self, is_debug):
        self.claims = {}
        self.labels = {}
        self.sentence_evidence = {}
        self.claims_parsed = {}
        self.claims_parsed_hierarchy = {}
        self.alignments = {}
        self.proofver_proofs = {}

        with open(self.save_path, "r") as f_in:
            lines = f_in.readlines()
            cutoff = len(lines) + 1 if not is_debug else 10
            for line in lines[:cutoff]:
                content = json.loads(line)
                qid = content["id"]
                self.claims[qid] = content["claim"]
                self.labels[qid] = content["verdict"]
                self.sentence_evidence[qid] = content["evidence"]
                self.proofver_proofs[qid] = content["proof"]
                self.claims_parsed[qid] = content["claim_parsed"]
                self.claims_parsed_hierarchy[qid] = content["claim_parsed_hierarchy"]
                self.alignments[qid] = content["alignment"]

    def setup_dataset(self):
        if self.dataset == "fever":
            from pyserini.search.lucene import LuceneSearcher

            self.dataset_reader = FeverReader(self.split, self.is_few_shot)
            self.proofver_proofs = self.dataset_reader.read_proofver_proofs()

            index_path = os.path.join(ROOT_DIR, "index", "lucene-index-{}-sentences-script").format(self.dataset)

            if os.path.exists(index_path):
                self.searcher_sentences = LuceneSearcher(index_path)  # SimpleSearcher(self.index)
                gold_evidences = {}
            else:
                print(
                    "Index Path to index for retrieving evidence not found. Make sure you download the index as described in the repository at: https://github.com/Raldir/QA-NatVer"
                )
                sys.exit()

            for i, anno in enumerate(self.dataset_reader.read_annotations()):
                qid, query, label, evidences = anno
                self.claims[qid] = query
                self.labels[qid] = label
                gold_evidences[qid] = evidences[0]

            self._load_evidence(gold_evidences)

        elif self.dataset == "danfever":
            self.dataset_reader = DanFeverReader(self.split, self.is_few_shot)
            self.searcher_sentences = LuceneSearcher(
                os.path.join(ROOT_DIR, "index", "lucene-index-{}-sentences-script").format(self.dataset)
            )  # SimpleSearcher(self.index)
            self.proofver_proofs = {}
            gold_evidences = {}
            for i, anno in enumerate(self.dataset_reader.read_annotations()):
                qid, query, label, evidences = anno
                qid = int(qid)
                self.claims[qid] = query
                self.labels[qid] = label
                gold_evidences[qid] = evidences
                self.proofver_proofs[qid] = []

            self._load_evidence(gold_evidences, already_filled=True)

            self._load_evidence(gold_evidences)

        # Make training data unique given multiple proofs
        if self.is_few_shot and "train" in self.split_name:
            new_claims = {}
            new_labels = {}
            new_sentence_evidence = {}
            new_proofs = {}

            label_stats = {"SUPPORTS": [], "REFUTES": [], "NOT ENOUGH INFO": []}
            num_per_class = int(self.num_samples / 3)
            num_per_class_ref = num_per_class + (self.num_samples - (3 * num_per_class))

            for key, proofs in self.proofver_proofs.items():
                label_stats[self.labels[key]].append(key)

            print({x: len(y) for x, y in label_stats.items()})

            selected_keys = set([])
            if self.stratified_sampling:
                supp_list = random.sample(self.label_stats["SUPPORTS"], num_per_class)
                nei_list = random.sample(self.label_stats["NOT ENOUGH INFO"], num_per_class)
                neg_list = random.sample(self.label_stats["REFUTES"], num_per_class + num_per_class_ref)
                selected_keys |= set(supp_list)
                selected_keys |= set(nei_list)
                selected_keys |= set(neg_list)
            else:
                all_keys = random.sample(list(self.proofver_proofs.keys()), self.num_samples)
                selected_keys = set(all_keys)

            assert (
                len(selected_keys) == self.num_samples
            ), "Selected samples and requested samples do not match, expected {} got {}".format(
                self.num_samples, len(selected_keys)
            )
            for key, proofs in self.proofver_proofs.items():
                if key not in selected_keys:
                    continue
                for i, proof in enumerate(proofs):
                    add_to_id = int(str(key) + ("0000000000" + str(i)))
                    new_claims[add_to_id] = self.claims[key]
                    new_labels[add_to_id] = self.labels[key]
                    new_sentence_evidence[add_to_id] = self.sentence_evidence[key]
                    new_proofs[add_to_id] = proof

            self.claims = new_claims
            self.labels = new_labels
            self.sentence_evidence = new_sentence_evidence
            self.proofver_proofs = new_proofs

            label_stats = {"SUPPORTS": 0, "REFUTES": 0, "NOT ENOUGH INFO": 0}
            all_keys = set([])
            for key, lab in self.labels.items():
                acc_key = str(key).split("0000000000")[0]
                if acc_key in all_keys:
                    continue
                all_keys.add(acc_key)
                label_stats[lab] += 1
            print("Label Distributions: ", label_stats)


if __name__ == "__main__":
    dataset = sys.argv[1]

    configs = sys.argv[2]

    # Add multiple configs via "+" "dynamic_simalign_bert_mwmf_coarse+dynamic_simalign_bert_mwmf
    configs = configs.split("+")

    configs_path = os.path.join(ROOT_DIR, "configs", "alignment")

    onlyfiles = [
        os.path.join(configs_path, f)
        for f in os.listdir(configs_path)
        if os.path.isfile(os.path.join(configs_path, f))
    ]
    onlyfiles = sorted(onlyfiles)

    for config_file in onlyfiles:
        config_file_name = config_file.split("/")[-1].split(".")[0]
        if not config_file_name in configs:
            continue
        with open(config_file, "r") as f_in:
            config = json.load(f_in)
        # for split in ["train"]:
        for split in ["validation"]:
            # for split in ["train", "validation"], "symmetric"]:
            dynamic_parsing = config["dynamic_parsing"]
            use_retrieved_evidence = config["use_retrieved_evidence"]
            num_retrieved_evidence = config["num_retrieved_evidence"]
            few_shot = config["few_shot"]
            max_chunks = config["max_chunks"]
            alignment_mode = config["alignment_mode"]
            alignment_model = config["alignment_model"]
            matching_method = config["matching_method"]
            loose_matching = config["loose_matching"]
            num_samples = 32  # Fixing samples to 32

            split_name = split + "-fewshot" if few_shot and "train" in split else split

            fever_data = DatasetProcessor(
                dataset=dataset,
                split=split,
                num_samples=num_samples,
                overwrite_data=True,
                dynamic_parsing=dynamic_parsing,
                is_few_shot=few_shot,
                use_retrieved_evidence=use_retrieved_evidence,
                num_retrieved_evidence=num_retrieved_evidence,
                ev_sentence_concat_op=" </s> ",
                max_chunks=max_chunks,
                alignment_mode=alignment_mode,
                alignment_model=alignment_model,
                matching_method=matching_method,
                loose_matching=loose_matching,
            )
            with open(
                "data/{}/processed_{}_num_samples_{}_use_retr_{}_retr_evidence_{}_dp_{}_alignment_mode_{}_max_chunks_{}_alignment_model_{}_matching_method_{}_loose_matching_{}.jsonl".format(
                    dataset,
                    split_name,
                    num_samples,
                    use_retrieved_evidence,
                    num_retrieved_evidence,
                    dynamic_parsing,
                    alignment_mode,
                    max_chunks,
                    alignment_model,
                    matching_method,
                    loose_matching,
                ),
                "w",
            ) as f_out:
                for key in fever_data.claims:
                    dictc = {}
                    if key in fever_data.proofver_proofs:
                        claim = fever_data.claims[key]
                        claims_parsed = fever_data.claims_parsed[key]
                        claim_parsed_hierarchy = fever_data.claims_parsed_hierarchy[key]
                        alignment = fever_data.alignments[key]
                        evidence = fever_data.sentence_evidence[key]
                        proof = fever_data.proofver_proofs[key]
                        verdict = fever_data.labels[key]
                        dictc["id"] = key
                        dictc["claim"] = claim
                        dictc["verdict"] = verdict
                        dictc["evidence"] = evidence
                        dictc["proof"] = proof
                        dictc["claim_parsed"] = claims_parsed
                        dictc["claim_parsed_hierarchy"] = claim_parsed_hierarchy
                        dictc["alignment"] = alignment
                        f_out.write("{}\n".format(json.dumps(dictc)))
                        print(claim)
                        print(claims_parsed)
                        print(alignment)
                        print(evidence)
                        print(proof)
                        print("--------")
