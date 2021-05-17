import torch
from torch.hub import load_state_dict_from_url


class LPIPS_reference(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg16 = load_state_dict_from_url(
            'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt',
            file_name='vgg16_stylegan.pth'
        )

    def forward(self, in0, in1):
        out0 = self.vgg16(in0, resize_images=False, return_lpips=True)
        out1 = self.vgg16(in1, resize_images=False, return_lpips=True)
        out = (out0 - out1).square().sum(dim=-1)
        return out
