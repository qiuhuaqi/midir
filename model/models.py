import torch.nn as nn
import numpy as np

from model.networks.dvf_nets import SiameseNet
from model.networks.ffd_nets import SiameseFFDNet, FFDNet
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
        self._set_transformation()

    def _set_network(self):
        if self.params.network == "SiameseNet":
            self.network = SiameseNet()

        elif self.params.network == "SiameseFFD":
            self.network = SiameseFFDNet()
        elif self.params.network == "FFDNet":
            self.network = FFDNet(self.params.dim,
                                  self.params.crop_size,
                                  self.params.ffd_cps)
        else:
            raise ValueError("Model: Network not recognised")

    def _set_transformation(self):
        if self.params.transformation == "DVF":
            self.transform = DVFTransform()

        elif self.params.transformation == "FFD":
            self.transform = BSplineFFDTransform(dim=self.params.dim,
                                                 img_size=self.params.crop_size,
                                                 cpt_spacing=self.params.ffd_cps)
        else:
            raise ValueError("Model: Transformation model not recognised")

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

