# NLPDV

### NOTE

- We perform efficient tranfer learning using Data Shapley method. 

### Requirements

python 3.6, [pytorch 1.3](https://pytorch.org/get-started/previous-versions/#commands-for-versions--100),  [tqdm](https://pypi.org/project/tqdm/), apex, huggingface transformers, 


### Baseline Runs
For Glue tasks, We consider MNLI, SNLI, QQP, QNLI tasks. We modify the [data processor](https://github.com/rizwan09/NLPDV/blob/master/transformers/src/transformers/data/processors/sglue.py) to convert the tasks into binary classification problem. Basleine perfromance means that we train a ```bert-base-cased``` model on all the datasets except the target and evaluate on the target dev set. In commit 3b99835 [bash script](https://github.com/rizwan09/NLPDV/blob/master/transformers/script_run_sglue_domain_binary_four_tasks.py) with modification (commenting/uncommenting) line 742 in the [code](https://github.com/rizwan09/NLPDV/blob/master/transformers/examples/run_sglue.py) will produce the result.

### Experiemnts
Compute the SEAL-Shap values for each dataset as follows: 
```
$ cd  NLPDV/transformers
$ python script_fair_pos_tagger.py
$ python script_run_overlapping_multi_domain_pos.py
$ python script_sXNLI.py
$ python script_run_overlapping_review_sentiment.py
$ python script_overlapping_sources_cached_ftmc_sglue.py

```

Then tune hyper-parameters after setting the filtered sources as follows:
```
$ cd  NLPDV/transformers
$ python script_tune_xnli.py
$ python script_tune_nli.py
$ python script_tune_review_sentiment.py
$ python run_flair.py
$ python script_tune_multi_pos.py

```



#### Running experiments on CPU/GPU/Multi-GPU

- Modify the gpu ids in CUDA_VISISBLE_DEVICES in [script_run_sglue.py](https://github.com/rizwan09/NLPDV/blob/master/transformers/script_run_sglue.py)

### Results

[Google Spreadsheet](https://docs.google.com/spreadsheets/d/1SE5wuhJtb070C--nbBrMWaj8JI0wwk0v2EiIqPMOfc4/edit?usp=sharing)



