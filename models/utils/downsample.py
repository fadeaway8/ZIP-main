from torch import nn, Tensor

from typing import Union

from .blocks import DepthSeparableConv2d, conv1x1, conv3x3
from .utils import _init_weights


class ConvDownsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Union[nn.BatchNorm2d, nn.GroupNorm, None] = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU(inplace=True),
        groups: int = 1,
    ) -> None:
        super().__init__()
        assert isinstance(groups, int) and groups > 0, f"Number of groups should be an integer greater than 0, but got {groups}."
        assert in_channels % groups == 0, f"Number of input channels {in_channels} should be divisible by number of groups {groups}."
        assert out_channels % groups == 0, f"Number of output channels {out_channels} should be divisible by number of groups {groups}."
        self.grouped_conv = groups > 1

        # conv1 is used for downsampling
        # self.conv1 = nn.Conv2d(
        #     in_channels=in_channels,
        #     out_channels=in_channels,
        #     kernel_size=2,
        #     stride=2,
        #     padding=0,
        #     bias=not norm_layer,
        #     groups=groups,
        # )
        # if self.grouped_conv:
        #     self.conv1_1x1 = conv1x1(in_channels, in_channels, stride=1, bias=not norm_layer)
        self.conv1 = nn.AvgPool2d(kernel_size=2, stride=2)  # downsample by 2
        if self.grouped_conv:
            self.conv1_1x1 = nn.Identity()
        
        self.norm1 = norm_layer(in_channels) if norm_layer else nn.Identity()
        self.act1 = activation

        self.conv2 = conv3x3(
            in_channels=in_channels,
            out_channels=in_channels,
            stride=1,
            groups=groups,
            bias=not norm_layer,
        )
        if self.grouped_conv:
            self.conv2_1x1 = conv1x1(in_channels, in_channels, stride=1, bias=not norm_layer)
        
        self.norm2 = norm_layer(in_channels) if norm_layer else nn.Identity()
        self.act2 = activation

        self.conv3 = conv3x3(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=1,
            groups=groups,
            bias=not norm_layer,
        )
        if self.grouped_conv:
            self.conv3_1x1 = conv1x1(out_channels, out_channels, stride=1, bias=not norm_layer)

        self.norm3 = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.act3 = activation

        self.downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),  # make sure the spatial sizes match
            conv1x1(in_channels, out_channels, stride=1, bias=not norm_layer),
            norm_layer(out_channels) if norm_layer else nn.Identity(),
        )
        
        self.apply(_init_weights)
    
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # downsample
        out = self.conv1(x)
        out = self.conv1_1x1(out) if self.grouped_conv else out
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.conv2_1x1(out) if self.grouped_conv else out
        out = self.norm2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.conv3_1x1(out) if self.grouped_conv else out
        out = self.norm3(out)

        # shortcut
        out += self.downsample(identity)
        out = self.act3(out)
        return out


class LightConvDownsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Union[nn.BatchNorm2d, nn.GroupNorm, None] = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU(inplace=True),
    ) -> None:
        super().__init__()
        self.conv1 = DepthSeparableConv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=not norm_layer,
        )
        self.norm1 = norm_layer(in_channels) if norm_layer else nn.Identity()
        self.act1 = activation

        self.conv2 = DepthSeparableConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not norm_layer,
        )
        self.norm2 = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.act2 = activation

        self.conv3 = DepthSeparableConv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not norm_layer,
        )
        self.norm3 = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.act3 = activation

        self.downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),  # make sure the spatial sizes match
            conv1x1(in_channels, out_channels, stride=1, bias=not norm_layer),
            norm_layer(out_channels) if norm_layer else nn.Identity(),
        )

        self.apply(_init_weights)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # downsample
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        # refine 1
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)

        # refine 2
        out = self.conv3(out)
        out = self.norm3(out)

        # shortcut
        out += self.downsample(identity)
        out = self.act3(out)
        return x
    

class LighterConvDownsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Union[nn.BatchNorm2d, nn.GroupNorm, None] = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU(inplace=True),
    ) -> None:
        super().__init__()
        self.conv1 = DepthSeparableConv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=not norm_layer,
        )
        self.norm1 = norm_layer(in_channels) if norm_layer else nn.Identity()
        self.act1 = activation

        self.conv2 = conv3x3(
            in_channels=in_channels,
            out_channels=in_channels,
            stride=1,
            groups=in_channels,
            bias=not norm_layer,
        )
        self.norm2 = norm_layer(in_channels) if norm_layer else nn.Identity()
        self.act2 = activation

        self.conv3 = conv1x1(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=1,
            bias=not norm_layer,
        )
        self.norm3 = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.act3 = activation

        self.downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),  # make sure the spatial sizes match
            conv1x1(in_channels, out_channels, stride=1, bias=not norm_layer),
            norm_layer(out_channels) if norm_layer else nn.Identity(),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # downsample
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        # refine, depthwise conv
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)

        # refine, pointwise conv
        out = self.conv3(out)
        out = self.norm3(out)

        # shortcut
        out += self.downsample(identity)
        out = self.act3(out)
        return out
