import json
import os

def main():
    in_file = os.path.join("data", "danfever", "processed_validation_use_retr_False_retr_evidence_2_dp_True_alignment_mode_awesomealign_max_chunks_6_alignment_model_bert_finetuned_default_gold_no_nei_few_shot_False_4000_matching_method_mwmf_loose_matching_False.jsonl")

    content = []
    content_new = []
    with open(in_file, "r") as f_in:
        lines = f_in.readlines()
        for line in lines:
            content = json.loads(line)
            new_dict = {}
            new_dict["id"] = content["id"]
            new_dict["claim"] = content["claim"]
            new_dict["evidence"] = content["evidence"]
            new_dict["verdict"] = content["verdict"]
            content_new.append(new_dict)

    out_file = os.path.join("data", "danfever", "danfever_validation.jsonl")
    with open(out_file, "w") as f_out:
        for element in content_new:
            f_out.write("{}\n".format(json.dumps(element)))


if __name__ == "__main__":
    main()