# ONION vs. MLM_Scoring Text Defense

* The code is based on the implementation of [HiddenKiller](https://github.com/thunlp/HiddenKiller), [ONION](https://github.com/thunlp/ONION) and [Masked Language Model Scoring](https://github.com/awslabs/mlm-scoring). 
* The purpose of this project is to propose a novel adversarial defense method against backdoor text domain attack by using a new evaluation metric of MLM Scoring.
* The code will be continously updated.

## LICENSES
- Includes other software related under the MIT and Apache 2.0 license
- ONION, Copyright 2021 THUNLP. For licensing see LICENSE-ONION
- mlm-scoring, Copyright Amazon.com, Inc. For licensing see LICENSE-mlm-scoring

## 1. Train Poisoned Victom Model (BERT)

* Train poisoned BERT for "SST-2":

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/run_poison_bert.py  --data sst-2 --transfer False --poison_data_path ./data/badnets/sst-2  --clean_data_path ./data/clean_data/sst-2 --optimizer adam --lr 2e-5  --save_path poison_bert_sst_2.pkl
```

* Train poisoned BERT for "Offenseval":

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/run_poison_bert.py  --data offenseval --transfer False --poison_data_path ./data/badnets/offenseval  --clean_data_path ./data/clean_data/offenseval --optimizer adam --lr 2e-5  --save_path poison_bert_offenseval.pkl
```

* Train poisoned BERT for "AG News":

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/run_poison_bert.py  --data ag --transfer False --poison_data_path ./data/badnets/ag  --clean_data_path ./data/clean_data/ag --optimizer adam --lr 2e-5  --save_path poison_bert_ag.pkl
```

* Train poisoned BERT for "DBPedia":

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/run_poison_bert.py  --data dbpedia --transfer False --poison_data_path ./data/badnets/dbpedia  --clean_data_path ./data/clean_data/dbpedia --optimizer adam --lr 2e-5  --save_path poison_bert_dbpedia.pkl
```

## 2. Defense Methods
### 2.1 Test Defense (ONION)

* Original ONION defense on "SST-2" against BadNets:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/test_defense.py  --data sst-2 --model_path poison_bert_sst_2.pkl  --poison_data_path ./data/badnets/sst-2/test.tsv  --clean_data_path ./data/clean_data/sst-2/dev.tsv
```

* Original ONION defense on "Offenseval" against BadNets:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/test_defense.py  --data offenseval --model_path poison_bert_offenseval.pkl  --poison_data_path ./data/badnets/offenseval/test.tsv  --clean_data_path ./data/clean_data/offenseval/dev.tsv
```

* Original ONION defense on "AG News" against BadNets:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/test_defense.py  --data ag --model_path poison_bert_ag.pkl  --poison_data_path ./data/badnets/ag/test.tsv  --clean_data_path ./data/clean_data/ag/dev.tsv
```

* Original ONION defense on "DBPedia" against BadNets:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/test_defense.py  --data dbpedia --model_path poison_bert_dbpedia.pkl  --poison_data_path ./data/badnets/dbpedia/test.tsv  --clean_data_path ./data/clean_data/dbpedia/dev.tsv
```


### 2.2 Test Defense (MLM_Scoring)

* First download required package:

```bash
pip install -e .
pip install torch mxnet
pip install mxnet-cu112
```

* MLM_Scoring Defense on "SST-2" against BadNets:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/test_defense_v3.py  --data sst-2 --model_path poison_bert_sst_2.pkl  --poison_data_path ./data/badnets/sst-2/test.tsv  --clean_data_path ./data/clean_data/sst-2/dev.tsv --record_file defense_MLM_sst2.log
```

* MLM_Scoring Defense on "Offenseval" against BadNets:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/test_defense_v3.py  --data offenseval --model_path poison_bert_offenseval.pkl  --poison_data_path ./data/badnets/offenseval/test.tsv  --clean_data_path ./data/clean_data/offenseval/dev.tsv --record_file defense_MLM_offenseval.log
```

* MLM_Scoring Defense on "AG News" against BadNets:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/test_defense_v3.py  --data ag --model_path poison_bert_ag.pkl  --poison_data_path ./data/badnets/ag/test.tsv  --clean_data_path ./data/clean_data/ag/dev.tsv --record_file defense_MLM_ag.log
```

* MLM_Scoring Defense on "DBPedia" against BadNets:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/test_defense_v3.py  --data dbpedia --model_path poison_bert_dbpedia.pkl  --poison_data_path ./data/badnets/dbpedia/test.tsv  --clean_data_path ./data/clean_data/dbpedia/dev.tsv --record_file defense_MLM_dbpedia.log
```


