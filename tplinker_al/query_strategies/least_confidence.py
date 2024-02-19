import numpy as np
from .strategy import Strategy

class LeastConfidence(Strategy):
    def __init__(self, dataset):
        super(LeastConfidence, self).__init__(dataset)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        # print("unlabeldata: {}".format(len(unlabeled_data)))
        probs = self.predict_prob_max(unlabeled_data)
        # print("probs.size: {}".format(probs.size()))
        uncertainties = probs.resize_(len(unlabeled_data))
        # uncertainties = probs.max(1)[0]#(1000,)
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
