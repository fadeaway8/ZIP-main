from torch import nn, Tensor
from torch.nn import functional as F

from typing import Union
from functools import partial

from .utils import _init_weights
from .refine import ConvRefine, LightConvRefine, LighterConvRefine


class ConvUpsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        norm_layer: Union[nn.BatchNorm2d, nn.GroupNorm, None] = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU(inplace=True),
        groups: int = 1,
    ) -> None:
        super().__init__()
        assert scale_factor >= 1, f"Scale factor should be greater than or equal to 1, but got {scale_factor}"
        self.scale_factor = scale_factor
        self.upsample = partial(
            F.interpolate,
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
            antialias=False,
        ) if scale_factor > 1 else nn.Identity()

        self.refine = ConvRefine(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_layer=norm_layer,
            activation=activation,
            groups=groups,
        )

        self.apply(_init_weights)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample(x)
        x = self.refine(x)
        return x


class LightConvUpsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        norm_layer: Union[nn.BatchNorm2d, nn.GroupNorm, None] = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU(inplace=True),
    ) -> None:
        super().__init__()
        assert scale_factor >= 1, f"Scale factor should be greater than or equal to 1, but got {scale_factor}"
        self.scale_factor = scale_factor
        self.upsample = partial(
            F.interpolate,
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
            antialias=False,
        ) if scale_factor > 1 else nn.Identity()

        self.refine = LightConvRefine(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_layer=norm_layer,
            activation=activation,
        )

        self.apply(_init_weights)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample(x)
        x = self.refine(x)
        return x


class LighterConvUpsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        norm_layer: Union[nn.BatchNorm2d, nn.GroupNorm, None] = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU(inplace=True),
    ) -> None:
        super().__init__()
        assert scale_factor >= 1, f"Scale factor should be greater than or equal to 1, but got {scale_factor}"
        self.scale_factor = scale_factor
        self.upsample = partial(
            F.interpolate,
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
            antialias=False,
        ) if scale_factor > 1 else nn.Identity()

        self.refine = LighterConvRefine(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_layer=norm_layer,
            activation=activation,
        )

        self.apply(_init_weights)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample(x)
        x = self.refine(x)
        return x
