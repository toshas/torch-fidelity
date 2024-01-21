import os

import numpy as np
import torch

from torch_fidelity.generative_model_base import GenerativeModelBase
from torch_fidelity.helpers import vassert


class GenerativeModelONNX(GenerativeModelBase):
    def __init__(self, path_onnx, z_size, z_type, num_classes):
        """
        Wraps :obj:`ONNX<torch:torch.onnx>` generative model, implements the :class:`GenerativeModelBase` interface.

        Args:

            path_onnx (str): Path to a generative model in :obj:`ONNX<torch:torch.onnx>` format.

            z_size (int): Size of the noise dimension of the generative model (positive integer).

            z_type (str): Type of the noise used by the generative model (see :ref:`registry <Registry>` for a list of
                preregistered noise types, see :func:`register_noise_source` for registering a new noise type).

            num_classes (int): Number of classes used by a conditional generative model. Must return zero for
                unconditional models.
        """
        super().__init__()
        vassert(os.path.isfile(path_onnx), f'Model file not found at "{path_onnx}"')
        vassert(type(z_size) is int and z_size > 0, "z_size must be a positive integer")
        vassert(z_type in ("normal", "unit", "uniform_0_1"), f"z_type={z_type} not implemented")
        vassert(type(num_classes) is int and num_classes >= 0, "num_classes must be a non-negative integer")
        try:
            import onnxruntime
        except ImportError as e:
            # This message may be removed if onnxruntime becomes a unified package with embedded CUDA dependencies,
            # like for example pytorch
            print(
                "====================================================================================================\n"
                "Loading ONNX models in PyTorch requires ONNX runtime package, which we did not want to include in\n"
                "torch_fidelity package requirements.txt. The two relevant pip packages are:\n"
                " - onnxruntime       (pip install onnxruntime), or\n"
                " - onnxruntime-gpu   (pip install onnxruntime-gpu).\n"
                'If you choose to install "onnxruntime", you will be able to run inference on CPU only - this may be\n'
                'slow. With "onnxruntime-gpu" speed is not an issue, but at run time you might face CUDA toolkit\n'
                "versions incompatibility, which can only be resolved by recompiling onnxruntime-gpu from source.\n"
                "Alternatively, use calculate_metrics API and pass an instance of GenerativeModelBase as an input.\n"
                "===================================================================================================="
            )
            raise e
        self.ort_session = onnxruntime.InferenceSession(path_onnx)
        self.input_names = [a.name for a in self.ort_session.get_inputs()]
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

    @staticmethod
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def forward(self, *args):
        vassert(
            len(args) == len(self.input_names),
            f"Number of input arguments {len(args)} does not match ONNX model: {self.input_names}",
        )
        vassert(all(torch.is_tensor(a) for a in args), "All model inputs must be tensors")
        ort_input = {self.input_names[i]: self.to_numpy(args[i]) for i in range(len(args))}
        ort_output = self.ort_session.run(None, ort_input)
        ort_output = ort_output[0]
        vassert(isinstance(ort_output, np.ndarray), "Invalid output of ONNX model")
        out = torch.from_numpy(ort_output).to(device=args[0].device)
        return out
