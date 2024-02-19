import numpy as np
import torch
from torchvision import datasets

class Data:
    def __init__(self, train_data,valid_data):
        self.train_data = train_data
        self.test_data = valid_data
        # self.handler = handler
        
        self.n_pool = len(train_data)
        self.n_test = len(valid_data)
        
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        
    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True
    
    def get_labeled_data(self):
        labeled_idxs= np.arange(self.n_pool)[self.labeled_idxs]
        labeled_data=[]
        for i in labeled_idxs:
            l_data = self.train_data[i]
            labeled_data.append(l_data)
        return labeled_idxs, labeled_data
    
    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        unlabeled_data = []
        for j in unlabeled_idxs:
            u_data = self.train_data[j]
            unlabeled_data.append(u_data)
        return unlabeled_idxs, unlabeled_data

    def get_train_data(self):
        return self.labeled_idxs.copy(), self.train_data

