#!/usr/bin/env python
# coding: utf-8

import sys,os
sys.path.append(os.getcwd())
import json
import os
from common.utils import Preprocessor, DefaultLogger
import wandb
import config
import argparse
import torch
from utils import get_strategy
from data import Data
from pprint import pprint

config = config.train_config
hyper_parameters = config["hyper_parameters"]
# export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

# AL参数
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2333, help="random seed")
parser.add_argument('--n_init_labeled', type=int, default=254, help="number of init labeled samples")
parser.add_argument('--n_query', type=int, default=200, help="number of queries per round")
parser.add_argument('--n_round', type=int, default=190, help="number of rounds")
parser.add_argument('--strategy_name', type=str, default="BALDDropout",
                    choices=["RandomSampling",
                             "LeastConfidence",
                             "MarginSampling",
                             "EntropySampling",
                             "LeastConfidenceDropout",
                             "MarginSamplingDropout",
                             "EntropySamplingDropout",
                             "KMeansSampling",
                             "KCenterGreedy",
                             "BALDDropout",
                             "AdversarialBIM",
                             "AdversarialDeepFool"], help="query strategy")
args = parser.parse_args()
pprint(vars(args))
print()

# device
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_num"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# for reproductivity
torch.manual_seed(hyper_parameters["seed"])  # pytorch random seed
torch.backends.cudnn.deterministic = True


data_home = config["data_home"]
experiment_name = config["exp_name"]
train_data_path = os.path.join(data_home, experiment_name, config["train_data"])
valid_data_path = os.path.join(data_home, experiment_name, config["valid_data"])
rel2id_path = os.path.join(data_home, experiment_name, config["rel2id"])


if config["logger"] == "wandb":
    # init wandb
    wandb.init(project=experiment_name,
               name=config["run_name"],
               config=hyper_parameters  # Initialize config
               )

    wandb.config.note = config["note"]

    model_state_dict_dir = wandb.run.dir
    logger = wandb
else:
    logger = DefaultLogger(config["log_path"], experiment_name, config["run_name"], config["run_id"], hyper_parameters)
    model_state_dict_dir = config["path_to_save_model"]
    if not os.path.exists(model_state_dict_dir):
        os.makedirs(model_state_dict_dir)

# Load Data
train_data = json.load(open(train_data_path, "r", encoding="utf-8"))
valid_data = json.load(open(valid_data_path, "r", encoding="utf-8"))


dataset = Data(train_data, valid_data)  # load dataset
strategy = get_strategy(args.strategy_name)(dataset)  # load strategy

# start experiment
dataset.initialize_labels(args.n_init_labeled)
print(f"number of labeled pool: {args.n_init_labeled}")
print(f"number of unlabeled pool: {dataset.n_pool - args.n_init_labeled}")
print(f"number of testing pool: {dataset.n_test}")
print()
# round 0 accuracy
print("Round 0")
f1_0 = strategy.train_valid()
print(f"Round 0 testing accuracy: {f1_0}")

pred_score_list=[]
for rd in range(1, args.n_round + 1):
    print(f"Round {rd}")

    # query
    query_idxs = strategy.query(args.n_query)
    # print(type(query_idxs))

    # update labels
    strategy.update(query_idxs)
    f1 = strategy.train_valid()
    print(f"Round {rd}  testing accuracy: {f1}")



