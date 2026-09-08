import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List, Optional

from .csrnet import _csrnet, _csrnet_bn
from ..utils import _init_weights

EPS = 1e-6


class ContextualModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 512,
        scales: List[int] = [1, 2, 3, 6],
    ) -> None:
        super().__init__()
        self.scales = scales
        self.multiscale_modules = nn.ModuleList([self.__make_scale__(in_channels, size) for size in scales])
        self.bottleneck = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.weight_net = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.apply(_init_weights)

    def __make_weight__(self, feature: Tensor, scale_feature: Tensor) -> Tensor:
        weight_feature = feature - scale_feature
        weight_feature = self.weight_net(weight_feature)
        return F.sigmoid(weight_feature)
    
    def __make_scale__(self, channels: int, size: int) -> nn.Module:
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(size, size)),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, feature: Tensor) -> Tensor:
        h, w = feature.shape[-2:]
        multiscale_feats = [F.interpolate(input=scale(feature), size=(h, w), mode="bilinear") for scale in self.multiscale_modules]
        weights = [self.__make_weight__(feature, scale_feature) for scale_feature in multiscale_feats]
        multiscale_feats = sum([multiscale_feats[i] * weights[i] for i in range(len(weights))]) / (sum(weights) + EPS)
        overall_features = torch.cat([multiscale_feats, feature], dim=1)
        overall_features = self.bottleneck(overall_features)
        overall_features = self.relu(overall_features)
        return overall_features


class CANNet(nn.Module):
    def __init__(
        self,
        model_name: str,
        block_size: Optional[int] = None,
        norm: str = "none",
        act: str = "none",
        scales: List[int] = [1, 2, 3, 6],
    ) -> None:
        super().__init__()
        assert model_name in ["csrnet", "csrnet_bn"], f"Model name should be one of ['csrnet', 'csrnet_bn'], but got {model_name}."
        assert block_size is None or block_size in [8, 16, 32], f"block_size should be one of [8, 16, 32], but got {block_size}."
        assert isinstance(scales, (tuple, list)), f"scales should be a list or tuple, got {type(scales)}."
        assert len(scales) > 0, f"Expected at least one size, got {len(scales)}."
        assert all([isinstance(size, int) for size in scales]), f"Expected all size to be int, got {scales}."
        self.model_name = model_name
        self.scales = scales

        csrnet = _csrnet(block_size=block_size, norm=norm, act=act) if model_name == "csrnet" else _csrnet_bn(block_size=block_size, norm=norm, act=act)
        self.block_size = csrnet.block_size

        self.encoder = csrnet.encoder
        self.encoder_channels = csrnet.encoder_channels
        self.encoder_reduction = csrnet.encoder_reduction  # feature map size compared to input size

        self.refiner = nn.Sequential(
            csrnet.refiner,
            ContextualModule(csrnet.refine_channels, 512, scales)
        )
        self.refiner_channels = 512
        self.refiner_reduction = csrnet.refiner_reduction  # feature map size compared to input size

        self.decoder = csrnet.decoder
        self.decoder_channels = csrnet.decoder_channels
        self.decoder_reduction = csrnet.decoder_reduction

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)
    
    def refine(self, x: Tensor) -> Tensor:
        return self.refiner(x)
    
    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encode(x)
        x = self.refine(x)
        x = self.decode(x)
        return x


def _cannet(block_size: Optional[int] = None, norm: str = "none", act: str = "none", scales: List[int] = [1, 2, 3, 6]) -> CANNet:
    return CANNet("csrnet", block_size=block_size, norm=norm, act=act, scales=scales)

def _cannet_bn(block_size: Optional[int] = None, norm: str = "none", act: str = "none", scales: List[int] = [1, 2, 3, 6]) -> CANNet:
    return CANNet("csrnet_bn", block_size=block_size, norm=norm, act=act, scales=scales)
