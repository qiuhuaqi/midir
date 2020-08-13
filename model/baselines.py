import torch



class _BaselineModel(object):
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, tar, src):
        return NotImplementedError


class Identity(_BaselineModel):
    def __init__(self, dim):
        super(Identity, self).__init__(dim)

    def __call__(self, tar, src):
        # identity DVF (N, 2, H, W) or (1, 3, H, W, D)
        img_size = tar.size()
        dvf = torch.zeros((img_size[0], self.dim, *img_size[2:])).type_as(tar)
        return dvf


class MirtkFFD(_BaselineModel):
    def __init__(self, dim):
        super(MirtkFFD, self).__init__(dim)

    def __call__(self, tar, src):
        dvf = None
        return dvf


class AntsSyn(_BaselineModel):
    def __init__(self, dim):
        super(AntsSyn, self).__init__(dim)

    def __call__(self, tar, src):
        # dvf = ants_syn_reg(tar, src)
        dvf = None
        return dvf
