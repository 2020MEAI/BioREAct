import numpy as np
from .strategy import Strategy

class MarginSampling(Strategy):
    def __init__(self, dataset):
        super(MarginSampling, self).__init__(dataset)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs1,probs2,probs3 = self.predict_prob(unlabeled_data)
        # uncertainties = probs.resize_(len(unlabeled_data))
        probs_sorted_1, idxs = probs1.sort(descending=True)
        probs_sorted_2, idxs = probs2.sort(descending=True)
        probs_sorted_3, idxs = probs3.sort(descending=True)
        uncertainties1 = probs_sorted_1[:,:, 0] - probs_sorted_1[:,:,1]
        uncertainties_1 = uncertainties1.sum(1)
        uncertainties2 = probs_sorted_2[:,:,:, 0] - probs_sorted_2[:,:,:,1]
        uncertainties2_1 = uncertainties2.sum(2)
        uncertainties_2 = uncertainties2_1.max(1)[0]
        uncertainties3 = probs_sorted_3[:, :, :, 0] - probs_sorted_3[:, :, :, 1]
        uncertainties3_1 = uncertainties3.sum(2)
        uncertainties_3 = uncertainties3_1.max(1)[0]
        uncertainties = uncertainties_1 + uncertainties_2 + uncertainties_3
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
