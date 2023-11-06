for seed in 42 1024 0 1 32
    do
        python3 -m src.train -c configs/alignment/$1.json+configs/dataset/$2.json+configs/environment/$3.json+configs/model/$4.json+configs/samples/$5.json -k exp_name=$2/environment_$3_model_$4_alignment_$1_samples_$5/${seed} seed=${seed}
    done

python3 -m src.utils.combine_results $2 environment_$3_model_$4_alignment_$1_samples_$5 best No
python3 -m src.utils.combine_results $2 environment_$3_model_$4_alignment_$1_samples_$5 last No