# BioREAct

**BioREAc: A Joint Biomedical Relation Extraction Model based on Active Learning.**

This code is based on this model: **[TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking](https://www.aclweb.org/anthology/2020.coling-main.138.pdf).** And the strategies of active learning mainly refer to toolkits: [DeepAL](https://github.com/ej0cl6/deep-active-learning)

BioREAct is a framework for biomedical entity relation joint extraction based on  six active learning strategies. It realizes the iterative active learning sample selection and the training process of the entity relation joint extraction model.
## Model Architecture

![Model](model.png) 

**Note: Please refer to Q&A and closed issues to find your question before proposed a new issue.**


## Usage
### Prerequisites
Our experiments are conducted on Python 3.6 and Pytorch == 1.6.0. 
The main requirements are:
* tqdm
* glove-python-binary==0.2.0
* transformers==3.0.2
* wandb # for logging the results
* yaml

In the root directory, run
```bash
pip install -e .
```

### Train
Set configuration in `tplinker_al/config.py` as follows:
```python
common["exp_name"] = genedsyn # genedsyn, CPI, DDI
common["device_num"] = 0 # 1, 2, 3 ...
common["encoder"] = "BERT" # BiLSTM
train_config["hyper_parameters"]["batch_size"] = 24 # 6 for webnlg and webnlg_star
train_config["hyper_parameters"]["match_pattern"] = "only_head_text" # "only_head_text" for webnlg_star and nyt_star; "whole_text" for webnlg and nyt.

# if the encoder is set to BioBERT
bert_config["pretrained_model_path"] = ""../pretrained_models/biobert-base-cased""

# Leave the rest as default
```

Start training
```
cd tplinker_al
python train_AL.py
```


### Evaluation
Set configuration in `tplinker_al/config.py` as follows:
```python

eval_config["model_state_dict_dir"] = "./wandb" # if use wandb, set "./wandb"; if you use default logger, set "./default_log_dir" 
eval_config["run_ids"] = ["46qer3r9", ] # If you use default logger, run id is shown in the output and recorded in the log (see train_config["log_path"]); If you use wandb, it is logged on the platform, check the details of the running projects.
eval_config["last_k_model"] = 1 # only use the last k models in to output results
# Leave the rest as the same as the training
```
Start evaluation by running `tplinker_al/Evaluation.ipynb`

# Reference
```
@inproceedings{wang-etal-2020-tplinker,
    title = "{TPL}inker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking",
    author = "Wang, Yucheng and Yu, Bowen and Zhang, Yueyang and Liu, Tingwen and Zhu, Hongsong and Sun, Limin",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.138",
    pages = "1572--1582"
}

@article{huang2021deepal,
  title={Deepal: Deep active learning in python},
  author={Huang, Kuan-Hao},
  journal={arXiv preprint arXiv:2111.15258},
  year={2021}
}
```

