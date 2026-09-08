from torch import nn, Tensor
from typing import Union

from .utils import _init_weights
from .blocks import BasicBlock, LightBasicBlock, conv1x1, conv3x3


class ConvRefine(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Union[nn.BatchNorm2d, nn.GroupNorm, None] = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU(inplace=True),
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.refine = BasicBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_layer=norm_layer,
            activation=activation,
            groups=groups,
        )
        self.apply(_init_weights)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.refine(x)


class LightConvRefine(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Union[nn.BatchNorm2d, nn.GroupNorm, None] = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU(inplace=True),
    ) -> None:
        super().__init__()
        self.refine = LightBasicBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_layer=norm_layer,
            activation=activation,
        )
        self.apply(_init_weights)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.refine(x)


class LighterConvRefine(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Union[nn.BatchNorm2d, nn.GroupNorm, None] = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU(inplace=True),
    ) -> None:
        super().__init__()
        # depthwise separable convolution
        self.conv1 = conv3x3(
            in_channels=in_channels,
            out_channels=in_channels,
            stride=1,
            groups=in_channels,
            bias=not norm_layer,
        )
        self.norm1 = norm_layer(in_channels) if norm_layer else nn.Identity()
        self.act1 = activation

        self.conv2 = conv1x1(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=1,
            bias=not norm_layer,
        )
        self.norm2 = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.act2 = activation

        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride=1, bias=not norm_layer),
                norm_layer(out_channels) if norm_layer else nn.Identity(),
            )
        else:
            self.downsample = nn.Identity()

        self.apply(_init_weights)
    
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += self.downsample(identity)
        out = self.act2(out)
        return out
