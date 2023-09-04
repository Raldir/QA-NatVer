### Installation

Setup a new conda environment, e.g. `python3.9` (tested only Python version 3.9)

Follow requirements.txt (hopefully, have not checked it is updated.)

Download Java:
https://jdk.java.net/19/

Set Java Path:
export JAVA_HOME=/home/bw447/jdk-19.0.2/

#### Downloading data

Download the FEVER dataset into data/fever:

```
wget https://fever.ai/download/fever/train.jsonl -P data/fever/
wget https://fever.ai/download/fever/shared_task_dev.jsonl -P data/fever/
```


#### Download alignment model

Either download the alignment model and place it in ```models/awesomealign``` from here:

`https://drive.google.com/file/d/1-391W3NKTlJLMpjUwjCv03H1Qk0ygWzC/view?usp=sharing`

or simply use the non-finetuned alignment model, by calling the config file `dynamic_awesomealign_bert_mwmf_coarse_only_retrieval_5_ev` when running FEVER below.


### Run FEVER

Note if you want to run the preprocessing (chunking and alignment pipeline yourself), you need to add the actual fever data into the appropriate data folder (see previous step). You might further see an error after preprocessing the training data, before it processes the validation data. That's ok, just run the program again. This is a bug that needs to be fixed.


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
