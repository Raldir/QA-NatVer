#!/bin/bash

# python3 -m src.train \
# 	-c configs/alignment/$1.json+configs/dataset/$2.json+configs/environment/$3.json+configs/model/$4.json+configs/samples/$5.json \
# 	-k exp_name=$2/environment_$3_model_$4_alignment_$1_samples_$5/$6 \
# 	seed=$6


ALIGNMENT="configs/alignment/dynamic_awesomealign_bert_mwmf_coarse_finetuned_4000_gold_no_nei_no_retrieval_2_ev.json"
DATASET="configs/dataset/climate-fever.json"
ENVIRONMENT="configs/environment/local.json"
MODEL="configs/model/bart0.json"
SAMPLES="configs/samples/64.json"

SEED=24601


python -m src.train \
	-c ${ALIGNMENT}+${DATASET}+${ENVIRONMENT}+${MODEL}+${SAMPLES} \
	-k exp_name=climate_fever_test \
	seed=${SEED}