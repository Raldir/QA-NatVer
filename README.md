### Installation

Follow requirements.txt (hopefully, have not checked it is updated.)

Note if you want to run the preprocessing (chunking and alignment pipeline yourself), you need to add the actual fever data into the appropriate data folder. You will see an error anyways, telling you that the path was not found.


### Run FEVER

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


### Run DanFEVER (not tested yet)

With arguments being the processed data, the dataset, the environment, the model, the sample size, and the seed.

```
./bin/run_few_shot.sh dynamic_awesomealign_bert_mwmf_coarse_finetuned_4000_gold_no_nei_no_retrieval_2_ev danfever local mt0_3b 32 42
```
