import torch.nn as nn

from torch_fidelity.helpers import vassert


class SampleSimilarityBase(nn.Module):
    def __init__(self, name):
        """
        Base class for samples similarity measures that can be used in :func:`calculate_metrics`.

        Args:

            name (str): Unique name of the subclassed sample similarity measure, must be the same as used in
                :func:`register_sample_similarity`.
        """
        super(SampleSimilarityBase, self).__init__()
        vassert(type(name) is str, "Sample similarity name must be a string")
        self.name = name

    def get_name(self):
        return self.name

    def forward(self, *args):
        """
        Returns the value of sample similarity between the inputs.
        """
        raise NotImplementedError
