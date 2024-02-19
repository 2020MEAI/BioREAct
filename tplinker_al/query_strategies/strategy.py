import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys,os
sys.path.append(os.getcwd()) # 将整个项目添加到搜索目录中
# from tplinker_al.train import al_train,predict_prob,predict_prob_dropout,predict_prob_dropout_split,get_embeddings,predict_prob_entropy
from train import al_train, predict_prob, predict_prob_dropout, predict_prob_dropout_split, get_embeddings, predict_prob_entropy, predict_prob_max


# from tplinker import train
# from ..train import al_train,predict_prob,predict_prob_dropout,predict_prob_dropout_split,get_embeddings

class Strategy:
    def __init__(self, dataset):
        self.dataset = dataset
        # self.net = net
        # self.train_AL= train_AL
        # self.probs = 0

    def query(self, n):
        pass

    def update(self, pos_idxs, neg_idxs=None):
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    def train_valid(self):
        labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        print(len(labeled_data))
        valid_data = self.dataset.test_data
        f1 = al_train(labeled_data, valid_data)
        return f1

    def predict_prob(self, data):
        probs1,probs2,probs3 = predict_prob(data)
        return probs1,probs2,probs3

    def predict_prob_max(self, data):
        probs = predict_prob_max(data)
        return probs

    def predict_prob_entropy(self, data):
        probs = predict_prob_entropy(data)
        return probs

    def predict_prob_dropout(self, data, n_drop=10):
        probs = predict_prob_dropout(data, n_drop=n_drop)
        # self.probs /= n_drop
        return probs

    def predict_prob_dropout_split(self, data, n_drop=10):
        probs = predict_prob_dropout_split(data, n_drop=n_drop)
        return probs
    
    def get_embeddings(self, data):
        embeddings = get_embeddings(data)
        return embeddings

