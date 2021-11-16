import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score, precision_score
import logging

class Evaluator:
    def __init__(self):
        self.count = 0 # count times of accumulation
        self.metrics : dict = {} # save metrics

    def accumulate(self, preds : torch.Tensor, labels : torch.Tensor) -> dict:
        '''Accumulate metrics'''
        raise NotImplementedError

    def clear(self):
        '''Clear metrics and count'''
        self.count = 0
        self.metrics.clear()

    def log_metrics(self):
        '''Print metrics using logging.info'''
        raise NotImplementedError

class Classification_Evaluator(Evaluator):
    '''Evaluator for classification task'''
    def __init__(self, true_label):
        super(Classification_Evaluator, self).__init__()
        self.true_label = true_label
        self.labels = []
        self.preds = []

    @torch.no_grad()
    def accumulate(self, preds : torch.Tensor, labels : torch.Tensor):
        assert preds.shape == labels.shape, \
            f"preds {preds.shape} has different shape compared with labels {labels.shape}"
        self.preds += list(preds.detach().cpu().numpy())
        self.labels += list(labels.detach().cpu().numpy())

    def log_metrics(self):
        args = {
            "y_pred" : np.array(self.preds),
            "y_true" : np.array(self.labels),
            "average" : "macro",
            "labels" : self.true_label,
            "zero_division" : 0
        }
        Precision = round(precision_score(**args), 4)
        Recall = round(recall_score(**args), 4)
        F1_score = round(f1_score(**args), 4)
        logging.info(f"Precision: {Precision}    Recall: {Recall}    F1_score: {F1_score}")