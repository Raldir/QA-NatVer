### Installation

Setup a new conda environment, e.g. `python3.9` (tested only Python version 3.9)

Then install QA-NatVer and all relevant dependencies:
```bash
python3 -m pip install -e .
```

Finally, download Java and place it into the root path of the repository:
https://jdk.java.net/19/

Set Java Path:
export JAVA_HOME=$PWD/QA-NatVer/jdk-19.0.2/

#### Downloading data

Download the FEVER dataset into data/fever:

```
wget https://fever.ai/download/fever/train.jsonl -P data/fever/
wget https://fever.ai/download/fever/shared_task_dev.jsonl -P data/fever/
```


#### Download alignment model

Either download the alignment model and place it in ```models/awesomealign``` from here:

`https://drive.google.com/file/d/1-391W3NKTlJLMpjUwjCv03H1Qk0ygWzC/view?usp=sharing`

or use the non-finetuned alignment model, by calling the config file `dynamic_awesomealign_bert_mwmf_coarse_only_retrieval_5_ev` when running the training command below.

### Run FEVER

Note if you want to run the preprocessing (chunking and alignment pipeline yourself), you need to add the actual fever data into the appropriate data folder (see previous step). You might further see an error after preprocessing the training data, before it processes the validation data. That's ok, just run the program again. This is a bug that will be fixed.

To train a QA-NatVer model run the following command:

```bash
./bin/run_few_shot.sh dynamic_awesomealign_bert_mwmf_coarse_finetuned_4000_gold_no_nei_only_retrieval_5_ev fever local bart0 32 42
```

The arguments correspond to config files in `configs/`, as following:
1. alignment: `dynamic_awesomealign_bert_mwmf_coarse_finetuned_4000_gold_no_nei_only_retrieval_5_ev`
2. dataset: `fever`
3. environment: `local`
4. model: `bart0`
5. samples: `32`
6. seed: `42`

You can freely modify the hyperparameters in the respective config files to modify he training process. For instance instead of using BART0, you can train QA-NatVer with Flan-T5:

```bash
./bin/run_few_shot.sh dynamic_awesomealign_bert_mwmf_coarse_finetuned_4000_gold_no_nei_only_retrieval_5_ev fever local flant5_xl 32 42
```

 All available arguments can be found in `src/utils/Config.py`.
 

### Run DanFEVER (not tested yet)

With arguments being the processed data, the dataset, the environment, the model, the sample size, and the seed.

```
./bin/run_few_shot.sh dynamic_awesomealign_bert_mwmf_coarse_finetuned_4000_gold_no_nei_no_retrieval_2_ev danfever local mt0_3b 32 42
```
