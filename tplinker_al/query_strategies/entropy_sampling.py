import numpy as np
import torch
from .strategy import Strategy

class EntropySampling(Strategy):
    def __init__(self, dataset):
        super(EntropySampling, self).__init__(dataset)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob_entropy(unlabeled_data)
        uncertainties = probs.resize_(len(unlabeled_data))
        # log_probs = torch.log(probs)
        # uncertainties = (probs*log_probs).sum(1)
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
