import torch
import torch.nn as nn
import torchvision
from torch.hub import load_state_dict_from_url

PT_VGG16_BASE_URL = 'https://download.pytorch.org/models/vgg16-397923af.pth'
PT_LPIPS_VGG16_URL = 'https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-vgg16-lpips.pth'


class VGG16features(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=False)
        vgg_pretrained_features.load_state_dict(load_state_dict_from_url(PT_VGG16_BASE_URL))
        vgg_pretrained_features = vgg_pretrained_features.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        return h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3


def spatial_average(in_tens):
    return in_tens.mean([2, 3]).squeeze(1)


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


class NetLinLayer(nn.Module):
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class LPIPS_VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.chns = [64, 128, 256, 512, 512]
        self.L = len(self.chns)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=True)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=True)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=True)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=True)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=True)
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        state_dict = load_state_dict_from_url(PT_LPIPS_VGG16_URL, progress=True)
        self.load_state_dict(state_dict)
        self.net = VGG16features()
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    @staticmethod
    def normalize(x):
        assert x.dtype == torch.uint8
        # torchvision values in range [0,1] mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
        mean_rescaled = (1 + torch.tensor([-.030, -.088, -.188], device=x.device)[None, :, None, None]) * 255 / 2
        inv_std_rescaled = 2 / (torch.tensor([.458, .448, .450], device=x.device)[None, :, None, None] * 255)
        x = (x.float() - mean_rescaled) * inv_std_rescaled
        return x

    def forward(self, in0, in1):
        assert torch.is_tensor(in0) and torch.is_tensor(in1), 'Inputs must be torch tensors'
        assert in0.dim() == 4 and in0.shape[1] == 3 and in0.dtype == torch.uint8, 'Input 0 is not B x 3 x H x W @ uint8'
        assert in1.dim() == 4 and in1.shape[1] == 3 and in1.dtype == torch.uint8, 'Input 1 is not B x 3 x H x W @ uint8'
        in0_input = self.normalize(in0)
        in1_input = self.normalize(in1)

        outs0 = self.net.forward(in0_input)
        outs1 = self.net.forward(in1_input)

        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(self.lins[kk].model(diffs[kk])) for kk in range(self.L)]
        val = sum(res)
        return val
