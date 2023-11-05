import json
import os
import statistics
import sys

from src.utils.util import ROOT_DIR

def select_scores(scores_list):
    score = scores_list[-1]
    score = {key.replace("proof_", ""): value for key, value in score.items()}
    return score


if __name__ == "__main__":
    dataset = sys.argv[1]

    experiment_path = sys.argv[2]

    experiment_paths = os.path.join(ROOT_DIR, "exp_out", dataset, experiment_path)

    subfolders = [f.path for f in os.scandir(experiment_paths) if f.is_dir()]

    all_scores = []
    for folder in subfolders:
        in_path = "dev_scores_qanatver.json"
        scores_path = os.path.join(folder, in_path)
        with open(scores_path, "r") as f_in:
            scores = []
            for line in f_in:
                scores.append(json.loads(line.strip()))

            selected_scores = select_scores(scores)
            all_scores.append(selected_scores)

    avg_scores = {"accuracy": [], "recall": [], "precision": [], "f1": []}
    variances = {"accuracy": 0, "recall": 0, "precision": 0, "f1": 0}
    for selected_scores in all_scores:
        for key, value in selected_scores.items():
            avg_scores[key].append(value)
    for key, value in avg_scores.items():
        mean = statistics.mean(value)
        variance = statistics.stdev(value)
        avg_scores[key] = mean
        variances[key] = variance

    out_file_name = "avg_scores_qanatver.json"

    with open(os.path.join(experiment_paths, out_file_name), "w") as f_out:
        f_out.write("{}\n".format(json.dumps(avg_scores)))
        f_out.write("{}\n".format(json.dumps(variances)))

    # prisnt(subfolders)
