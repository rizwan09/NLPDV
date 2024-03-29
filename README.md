# Evaluating the Values of Sources in Transfer Learning
####  Md Rizwan Parvez, Kai-Wei Chang
##### NAACL 2021 (paper link: [Arxiv](https://arxiv.org/abs/2104.12567) or [ACL](https://www.aclweb.org/anthology/2021.naacl-main.402.pdf))
Transfer learning that adapts a model trained on data-rich sources to low-resource targets has been widely applied in natural language processing (NLP). However, when training a transfer model over multiple sources, not every source is equally useful for the target. To better transfer a model, it is essential to understand the values of the sources. In this paper, we develop SEAL-Shap, an efficient source valuation framework for quantifying the usefulness of the sources (e.g., domains/languages) in transfer learning based on the Shapley value method. Experiments and comprehensive analyses on both cross-domain and cross-lingual transfers demonstrate that our framework is not only effective in choosing useful transfer sources but also the source values match the intuitive source-target similarity.

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

