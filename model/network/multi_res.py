import torch
import torch.nn as nn

from model.network.base import conv_Nd
from model.network.single_res import UNet


class mlUNet(UNet):
    def __init__(self,
                 dim=2,
                 enc_channels=(16, 32, 32, 32, 32),
                 dec_channels=(32, 32, 32, 16),
                 ml_lvls=3
                 ):
        super(mlUNet, self).__init__(dim=dim,
                                     enc_channels=enc_channels,
                                     dec_channels=dec_channels,
                                     out_channels=None,
                                     conv_before_out=False)
        # number of multi-resolution levels
        self.ml_lvls = ml_lvls

        # conv layers for multi-resolution predictions (low res to high res)
        self.ml_pred_convs = nn.ModuleList()
        for l in range(self.ml_lvls):
            ml_pred_conv = conv_Nd(dim, enc_channels[self.ml_lvls - l - 1] + dec_channels[l - self.ml_lvls], dim)
            self.ml_pred_convs.append(ml_pred_conv)

    def forward(self, tar, src):
        x = torch.cat((tar, src), dim=1)

        # encoder
        fm_enc = [x]
        for enc in self.enc:
            fm_enc.append(enc(fm_enc[-1]))

        # decoder: conv + upsample + concatenate skip-connections (to full resolution)
        dec_out = fm_enc[-1]
        concat_out_ls = []
        for i, dec in enumerate(self.dec):
            dec_out = dec(dec_out)
            dec_out = self.upsample(dec_out)
            dec_out = torch.cat([dec_out, fm_enc[-2-i]], dim=1)
            concat_out_ls.append(dec_out)

        # construct multi-resolution dvf outputs
        ml_dvfs = []
        for l, pred_conv in enumerate(self.ml_pred_convs):
            pred_l = pred_conv(concat_out_ls[l-self.ml_lvls])
            if l == 0:
                # direct prediction for the coarsest level
                dvf_l = pred_l
            else:
                # predict residual and add to coarser level
                # TODO: return upsample and scaled dvf to compare before/after residual
                dvf_l = self.upsample(ml_dvfs[-1]) * 2.0 + pred_l
            ml_dvfs.append(dvf_l)

        assert len(ml_dvfs) == self.ml_lvls

        # # hardcode example (3 levels):
        # ml_dvf_lvl1 = self.ml_pred_convs[0](concat_out_ls[-3])
        #
        # ml_pred_lvl2 = self.ml_pred_convs[1](concat_out_ls[-2])
        # ml_dvf_lvl2 = self.upsample(ml_dvf_lvl1) + ml_pred_lvl2
        #
        # ml_pred_lvl3 = self.ml_pred_convs[2](concat_out_ls[-1])
        # ml_dvf_lvl3 = self.upsample(ml_pred_lvl2) + ml_pred_lvl3
        #
        # ml_dvfs = [ml_dvf_lvl1, ml_dvf_lvl2, ml_dvf_lvl3]

        return ml_dvfs
