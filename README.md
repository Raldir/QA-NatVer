### Run

With arguments being the processed data, the dataset, the environment, the model, the sample size, and the seed.

```
./bin/run_few_shot.sh dynamic_awesomealign_bert_mwmf_coarse_finetuned_4000_gold_no_nei_only_retrieval_5_ev fever local bart0 32 42
```

```
./bin/run_few_shot.sh dynamic_awesomealign_bert_mwmf_coarse_finetuned_4000_gold_no_nei_only_retrieval_5_ev fever local flant5_xl 32 42
```

or running over multiple seeds directly:

```
./bin/run_few_shot_seeds.sh dynamic_awesomealign_bert_mwmf_coarse_finetuned_4000_gold_no_nei_only_retrieval_5_ev fever local bart0 32
```
