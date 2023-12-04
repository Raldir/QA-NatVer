## Installation

Setup a new conda environment, e.g. `python3.9` (tested only Python version 3.9):

```bash
conda create -n qa_natver python=3.9
conda activate qa_natver
```

Then install QA-NatVer and all relevant dependencies:
```bash
python3 -m pip install -e .
```

## Training


### Preprocessing

The repository already contains the preprocessed, chunked, and aligned FEVER data both for the few-shot training and validation data, located at `data/fever`. 

However, if you want to preprocess data yourself, you need to execute the following steps:

#### FEVER Dataset

Download the FEVER dataset into data/fever:

```
wget https://fever.ai/download/fever/train.jsonl -P data/fever/
wget https://fever.ai/download/fever/shared_task_dev.jsonl -P data/fever/
```

#### Alignment model

Either download the alignment model and place it in ```models/awesomealign``` from here:

`https://drive.google.com/file/d/1-391W3NKTlJLMpjUwjCv03H1Qk0ygWzC/view?usp=sharing`

or use the non-finetuned alignment model, by calling the config file `dynamic_awesomealign_bert_mwmf_coarse_only_retrieval_5_ev` when running the training command below.

#### Retrieved data
The pipeline to incorporate retrieved data uses Pyserini. The dependencies are already installed, however you also need to download Java and place it into the root path of the repository:
https://jdk.java.net/19/

Set Java Path:
export JAVA_HOME=$PWD/QA-NatVer/jdk-19.0.2/

To use retrieved data, you need to add a file into `data/fever/retrieved_evidence` in the [pyserini format](https://github.com/castorini/pyserini), with the associated index. To create a Pyserini index, you first have to convert the json files into the right format (see `src/utils/create_pyserini_format.py`), and then create the index via Pyserini:

```bash
python3 -m pyserini.index.lucene --collection JsonCollection --input data/fever/sentences/ --index index/lucene-index-fever-sentences-script --generator DefaultLuceneDocumentGenerator --threads 1 --storePositions --storeDocvectors --storeRaw
```


Note if you want to run the preprocessing (chunking and alignment pipeline yourself), you need to add the actual fever data into the appropriate data folder (see previous step). You might further see an error after preprocessing the training data, before it processes the validation data. That's ok, just run the program again. This is a bug that will be fixed.

To train a QA-NatVer model run the following command:

```bash
./bin/train_few_shot.sh dynamic_awesomealign_bert_mwmf_coarse_finetuned_4000_gold_no_nei_only_retrieval_5_ev fever local bart0 32 0
```

The arguments correspond to config files in `configs/`, as following:
1. alignment: `dynamic_awesomealign_bert_mwmf_coarse_finetuned_4000_gold_no_nei_only_retrieval_5_ev`
2. dataset: `fever`
3. environment: `local`
4. model: `bart0`
5. samples: `32`
6. seed: `0`

You can freely modify the hyperparameters in the respective config files to modify he training process. For instance instead of using BART0, you can train QA-NatVer with Flan-T5:

```bash
./bin/run_few_shot.sh dynamic_awesomealign_bert_mwmf_coarse_finetuned_4000_gold_no_nei_only_retrieval_5_ev fever local flant5_xl 32 42
```

 All available arguments can be found in `src/utils/Config.py`.
 
## Inference

If you want to use QA-NatVer out of the box for existing claim evidence pairs, you can use existing models. There are currently two models available, trained on 32 FEVER samples: 

QA-NatVer with a BART0 backbone:
`https://drive.google.com/file/d/1fSDA7rlp39tEDAehoN8EeybEf9wSqEBo/view?usp=sharing`


QA-NatVer with a Flan-T5-xl backbone:
`https://drive.google.com/file/d/13o4iOTHeeDGqernuyi4f_2x5HTR20UFy/view?usp=sharing`


Download one of the QA-NatVer checkpoints and put `finish.pt` into `models/BART0`:

The input data should be a placed into `data/test.jsonl` file, with each line consisting of am `id`, `claim`, and an `evidence` field. The evidence field is expected to be a list of sentences. Look at the dummy data as found in the repository. The QA-NatVer with the Flan-T5 backbone will be made available soon.

To run the input data on QA-NatVer call:

```bash
./bin/run_few_shot.sh dynamic_awesomealign_bert_mwmf_coarse_finetuned_4000_gold_no_nei_only_retrieval_5_ev fever local_saving bart0_trained 32 0
```

## DanFEVER (Multilingual QA-NatVer)

We further provide a multilingual variation of QA-NatVer, trained on FEVER data with english questions on a multilingual backbone (mT0).

Download QA-NatVer with a mT0-3B backbone here:
`https://drive.google.com/file/d/1tviJ1tFsV2ERvDqHDhPFANlMFiKmstG4/view?usp=sharing`


Note that while the alignment system is multilingual (multilingual BERT), the chunking system is not. We use [flair](https://huggingface.co/flair/chunk-english) for chunking in English, and [DaNLP](https://danlp-alexandra.readthedocs.io/en/latest/) for chunking in Danish. To achieve best results on your target language, you would want to consider adjusting the chunking system.


Train and evaluate QA-NatVer on DanFEVER using the already processed data, with arguments being the processed data, the dataset, the environment, the model, the sample size, and the seed.

```
./bin/train_few_shot.sh dynamic_awesomealign_bert_mwmf_coarse_finetuned_4000_gold_no_nei_no_retrieval_2_ev danfever local_saving mt0_3b 32 0
```

TODO: Describe how to install and run DaNLP chunking.
