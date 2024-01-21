import copy

import torch

from torch_fidelity.generative_model_base import GenerativeModelBase
from torch_fidelity.helpers import vassert


class GenerativeModelModuleWrapper(GenerativeModelBase):
    def __init__(self, module, z_size, z_type, num_classes, make_copy=False, make_eval=True, cuda=None):
        """
        Wraps any generative model :class:`torch.nn.Module`, implements the :class:`GenerativeModelBase` interface, and
        provides a few convenience functions.

        Args:

            module (torch.nn.Module): A generative model module, taking a batch of noise samples, and producing
                generative samples.

            z_size (int): Size of the noise dimension of the generative model (positive integer).

            z_type (str): Type of the noise used by the generative model (see :ref:`registry <Registry>` for a list of
                preregistered noise types, see :func:`register_noise_source` for registering a new noise type).

            num_classes (int): Number of classes used by a conditional generative model. Must return zero for
                unconditional models.

            make_copy (bool): Makes a copy of the model weights if `True`. Default: `False`.

            make_eval (bool): Switches to :class:`torch.nn.Module` evaluation mode upon construction if `True`. Default:
                `True`.

            cuda (bool): Moves the module on a CUDA device if `True`, moves to CPU if `False`, does nothing if `None`.
                Default: `None`.
        """
        super().__init__()
        vassert(isinstance(module, torch.nn.Module), "Not an instance of torch.nn.Module")
        vassert(type(z_size) is int and z_size > 0, "z_size must be a positive integer")
        vassert(z_type in ("normal", "unit", "uniform_0_1"), f"z_type={z_type} not implemented")
        vassert(type(num_classes) is int and num_classes >= 0, "num_classes must be a non-negative integer")
        self.module = module
        if make_copy:
            self.module = copy.deepcopy(self.module)
        if make_eval:
            self.module.eval()
        if cuda is not None:
            if cuda:
                self.module = self.module.cuda()
            else:
                self.module = self.module.cpu()
        self._z_size = z_size
        self._z_type = z_type
        self._num_classes = num_classes

    @property
    def z_size(self):
        return self._z_size

    @property
    def z_type(self):
        return self._z_type

    @property
    def num_classes(self):
        return self._num_classes

    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)
