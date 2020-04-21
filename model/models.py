import torch
import torch.nn as nn
import numpy as np

from model.networks import BaseNet, SiameseFCN, BaseNetFFD
from model.transformations import BSplineFFDTransform, DVFTransform


class DLRegModel(nn.Module):
    def __init__(self, params):
        super(DLRegModel, self).__init__()

        self.params = params

        self.epoch_num = 0
        self.iter_num = 0
        self.is_best = False
        self.best_metric_result = 0

        self._set_network()
        self._set_transform_model()

    def _set_network(self):
        if self.params.network == "BaseNet":
            self.network = BaseNet()

        elif self.params.network == "BaseNetFFD":
            self.network = BaseNetFFD()

        elif self.params.network == "Siamese":
            self.network = SiameseFCN()

        else:
            raise ValueError("Network not recognised.")

    def _set_transform_model(self):
        if self.params.transform_model == "ffd":

            self.transform = BSplineFFDTransform(crop_size=self.params.crop_size,
                                                 cps=self.params.ffd_cps)

        elif self.params.transform_model == "dvf":
            self.transform = DVFTransform()

        else:
            raise ValueError("Transformation model not recognised.")

    def update_best_model(self, metric_results):
        metric_results_mean = np.mean([metric_results[metric] for metric in self.params.best_metrics])

        if self.epoch_num + 1 == self.params.val_epochs:
            # initialise for the first validation
            self.best_metric_result = metric_results_mean
        else:
            if metric_results_mean < self.best_metric_result:
                self.is_best = True
                self.best_metric_result = metric_results_mean

    def forward(self, target, source):
        net_out = self.network(target, source)
        dvf = self.transform(net_out)
        return dvf


"""
Baseline Models
"""
class BaselineModel(object):
    def __init__(self):
        pass

    def eval(self):
        # dummy method for model.eval() call in evaluate()
        pass

    def __call__(self, target, source):
        raise NotImplementedError


class IdBaselineModel(BaselineModel):
    """Identity transformation baseline, i.e. no registration"""
    def __init__(self, params):
        super(IdBaselineModel, self).__init__()
        self.params = params

    def __call__(self, target, source):
        """Output dvf in shape (N, dim, *(dims))"""
        dim = len(target.size())-2   # image dimension
        dvf = torch.zeros_like(target).repeat(1, dim, *(1,)*dim)
        return dvf

