import torch
import torch.nn as nn
from model.networks import BaseNet, SiameseFCN, BaseNetFFD
from model.submodules import spatial_transform
from model.transformation import BSplineFFDTransform, DVFTransform

"""General Spatial Transformer Network model"""
class RegDVF(nn.Module):
    def __init__(self, network_arch="BaseNet",
                 transform_model="ffd", cps=8):
        super().__init__()
        if network_arch == "BaseNet":
            self.network = BaseNet(ffd=transform_model=="ffd")
        elif network_arch == "BaseNetFFD":
            self.network = BaseNetFFD()
        elif network_arch == "Siamese":
            self.network = SiameseFCN()
        else:
            raise ValueError("Unknown network!")

        if transform_model == "ffd":
            self.transform_model = BSplineFFDTransform(cps=cps)
        elif transform_model == "dvf":
            self.transform_model = DVFTransform()
        else:
            raise ValueError("Unknown transformation model.")

    def forward(self, target, source):
        output = self.network(target, source)
        return self.transform_model(output, source)  # dvf, warped_source

