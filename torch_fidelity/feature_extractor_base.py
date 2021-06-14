import torch.nn as nn

from torch_fidelity.helpers import vassert


class FeatureExtractorBase(nn.Module):
    def __init__(self, name, features_list):
        """
        Base class for feature extractors that can be used in :func:`calculate_metrics`.

        Args:

            name (str): Unique name of the subclassed feature extractor, must be the same as used in
                :func:`register_feature_extractor`.

            features_list (list): List of feature names, provided by the subclassed feature extractor.
        """
        super(FeatureExtractorBase, self).__init__()
        vassert(type(name) is str, 'Feature extractor name must be a string')
        vassert(type(features_list) in (list, tuple), 'Wrong features list type')
        vassert(
            all((a in self.get_provided_features_list() for a in features_list)),
            f'Requested features {tuple(features_list)} are not on the list provided by the selected feature extractor '
            f'{self.get_provided_features_list()}'
        )
        vassert(len(features_list) == len(set(features_list)), 'Duplicate features requested')
        self.name = name
        self.features_list = features_list

    def get_name(self):
        return self.name

    @staticmethod
    def get_provided_features_list():
        """
        Returns a tuple of feature names, extracted by the subclassed feature extractor.
        """
        raise NotImplementedError

    def get_requested_features_list(self):
        return self.features_list

    def convert_features_tuple_to_dict(self, features):
        # The only compound return type of the forward function amenable to JIT tracing is tuple.
        # This function simply helps to recover the mapping.
        vassert(
            type(features) is tuple and len(features) == len(self.features_list),
            'Features must be the output of forward function'
        )
        return dict(((name, feature) for name, feature  in zip(self.features_list, features)))

    def forward(self, input):
        """
        Returns a tuple of tensors extracted from the `input`, in the same order as they are provided by
        `get_provided_features_list()`.
        """
        raise NotImplementedError
