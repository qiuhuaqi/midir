import torch


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