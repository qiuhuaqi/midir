import torch.nn as nn
import numpy as np

from model.networks.dvf_nets import SiameseNet
from model.networks.ffd_nets import SiameseFFDNet
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
        if self.params.network == "SiameseNet":
            self.network = SiameseNet()
        elif self.params.network == "SiameseFFD":
            self.network = SiameseFFDNet()
        else:
            raise ValueError("Network not recognised.")

    def _set_transform_model(self):
        if self.params.transform_model == "dvf":
            self.transform = DVFTransform()

        elif self.params.transform_model == "ffd":
            self.transform = BSplineFFDTransform(dim=2,
                                                 img_size=self.params.crop_size,
                                                 cpt_spacing=self.params.ffd_cps)
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

