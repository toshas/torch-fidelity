import torch.nn as nn

from torch_fidelity.helpers import vassert


class SampleSimilarityBase(nn.Module):
    def __init__(self, name):
        super(SampleSimilarityBase, self).__init__()
        vassert(type(name) is str, 'Sample similarity name must be a string')
        self.name = name

    def get_name(self):
        return self.name

    def forward(self, *args):
        raise NotImplementedError
