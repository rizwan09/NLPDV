# NLPDV

### NOTE

- We perform efficient tranfer learning using Data Shapley method. 
- Our Codebase added previous snapshots of [allennlp](https://github.com/allenai/allennlp), [Huggingface Transformers](https://github.com/huggingface/transformers) and [a previous Shapley value implemenattion](https://github.com/amiratag/DataShapley). 

### Requirements

python 3.6, [pytorch 1.3](https://pytorch.org/get-started/previous-versions/#commands-for-versions--100),  [tqdm](https://pypi.org/project/tqdm/), apex, huggingface transformers, 

### Compute Shapley Values and identifies the positive-negative source domains

Run the setup by running the [bash script](https://github.com/rizwan09/NLPDV/blob/master/transformers/script_run_sglue.py) as follows.

```
$ cd  NLPDV/transformers
$ python script_run_sglue.py
```

### To tune hyperparams or to report the final performance train on all positive source domains

Run the setup by running the [bash script](https://github.com/rizwan09/NLPDV/blob/master/transformers/script_run_sglue_domain_binary_four_tasks.py) as follows. So far the corresponding source domains for QNLI and MNLI-mismatched are hardcoded. 

```
$ cd  NLPDV/transformers
$ python script_run_sglue_domain_binary_four_tasks.py
```

### For Baseline POS tagger Perfromance

```
$ cd  NLPDV/transformers
$  CUDA_VISIBLE_DEVICES=7 python run_flair.py &>> log/weblogs_baseline.txt &
```


#### Running experiments on CPU/GPU/Multi-GPU

- Modify the gpu ids in CUDA_VISISBLE_DEVICES in [script_run_sglue.py](https://github.com/rizwan09/NLPDV/blob/master/transformers/script_run_sglue.py)

### Citation

```
@inproceedings{parvez2021evaluating,
  title = {Evaluating the Values of Sources in Transfer Learning},
  author = {Parvez, Md Rizwan and Chang, Kai-Wei},
  booktitle = {Proceedings of the 2021 Conference of the North {A}merican Chapter of the Association for Computational Linguistics},
  year = {2021}
}
```

