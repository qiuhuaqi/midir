import torch
import torch.nn as nn
from model.networks import BaseNet, SiameseFCN
from model.submodules import spatial_transform

"""General Spatial Transformer Network model"""
class RegDVF(nn.Module):
    def __init__(self, network_arch="BaseNet"):
        super().__init__()
        if network_arch == "BaseNet":
            self.network = BaseNet()
        elif network_arch == "Siamese":
            self.network = SiameseFCN()
        else:
            raise ValueError("Unknown network!")

    def forward(self, target, source):
        dvf = self.network(target, source)
        warped_source = spatial_transform(source, dvf)
        return dvf, warped_source

