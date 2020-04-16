import torch.nn as nn
import numpy as np
from model.networks import BaseNet, SiameseFCN, BaseNetFFD
from model.transform_models import BSplineFFDTransform, DVFTransform

class RegModel(nn.Module):
    def __init__(self, params):
        super(RegModel, self).__init__()

        self.params = params

        self._set_network()
        self._set_transform_model()

        self.epoch_num = 0
        self.iter_num = 0
        self.is_best = False
        self.best_metric_result = 0

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
