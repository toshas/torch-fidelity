from abc import ABC, abstractmethod

import torch


class GenerativeModelBase(ABC, torch.nn.Module):
    @property
    @abstractmethod
    def z_size(self):
        pass

    @property
    @abstractmethod
    def z_type(self):
        pass

    @property
    @abstractmethod
    def num_classes(self):
        pass
