# ONION vs. MLM_Scoring Backdoor Text Defense Method

* The code is highly based on the implementation of [ONION](https://github.com/thunlp/ONION) and [Masked Language Model Scoring](https://github.com/awslabs/mlm-scoring). 
* The purpose of this project is to propose a novel adversarial defence method against backdoor text domain attack by using a new evaluation metric of MLM Scoring.
* The code will be continously updated.

#### Includes other software related under the MIT and Apache 2.0 license:
- ONION, Copyright 2021 THUNLP. For licensing see LICENSE-ONION
- mlm-scoring, Copyright Amazon.com, Inc. For licensing see LICENSE-mlm-scoring

## 1. Train Poisoned Victom Model (BERT)

* Train poisoned BERT for SST-2:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/run_poison_bert.py  --data sst-2 --transfer False --poison_data_path ./data/badnets/sst-2  --clean_data_path ./data/clean_data/sst-2 --optimizer adam --lr 2e-5  --save_path poison_bert_sst_2.pkl
```

* Train poisoned BERT for Offenseval:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/run_poison_bert.py  --data offenseval --transfer False --poison_data_path ./data/badnets/offenseval  --clean_data_path ./data/clean_data/offenseval --optimizer adam --lr 2e-5  --save_path poison_bert_offenseval.pkl
```

## 2.1 Test Defense (ONION)

* Original ONION defense on SST-2 against BadNets:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/test_defense.py  --data sst-2 --model_path poison_bert_sst_2.pkl  --poison_data_path ./data/badnets/sst-2/test.tsv  --clean_data_path ./data/clean_data/sst-2/dev.tsv
```

* Original ONION defense on Offenseval against BadNets:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/test_defense.py  --data offenseval --model_path poison_bert_offenseval.pkl  --poison_data_path ./data/badnets/offenseval/test.tsv  --clean_data_path ./data/clean_data/offenseval/dev.tsv
```

## 2.2 Test Defense (MLM_Scoring)

* First download required package:

```bash
pip install -e .
pip install torch mxnet
pip install mxnet-cu112
```

* MLM_Scoring defense on SST-2 against BadNets:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/test_defense_v2.py  --data sst-2 --model_path poison_bert_sst_2.pkl  --poison_data_path ./data/badnets/sst-2/test.tsv  --clean_data_path ./data/clean_data/sst-2/dev.tsv --record_file defense_v2.log
```

* Original ONION defense on Offenseval against BadNets:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/test_defense_v2.py  --data offenseval --model_path poison_bert_offenseval.pkl  --poison_data_path ./data/badnets/offenseval/test.tsv  --clean_data_path ./data/clean_data/offenseval/dev.tsv --record_file defense_v2.log
```



