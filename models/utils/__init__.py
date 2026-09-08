from torch import nn
from typing import Optional
from functools import partial

from .utils import _init_weights, interpolate_pos_embed
from .blocks import DepthSeparableConv2d, conv1x1, conv3x3, Conv2dLayerNorm
from .refine import ConvRefine, LightConvRefine, LighterConvRefine
from .downsample import ConvDownsample, LightConvDownsample, LighterConvDownsample
from .upsample import ConvUpsample, LightConvUpsample, LighterConvUpsample
from .multi_scale import MultiScale
from .blocks import ConvAdapter, ViTAdapter


def _get_norm_layer(model: nn.Module) -> Optional[nn.Module]:
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            return nn.BatchNorm2d
        elif isinstance(module, nn.GroupNorm):
            num_groups = module.num_groups
            return partial(nn.GroupNorm, num_groups=num_groups)
        elif isinstance(module, (nn.LayerNorm, Conv2dLayerNorm)):
            return Conv2dLayerNorm
    return None


def _get_activation(model: nn.Module) -> Optional[nn.Module]:
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            return nn.ReLU(inplace=True)
        elif isinstance(module, nn.GroupNorm):
            return nn.ReLU(inplace=True)
        elif isinstance(module, (nn.LayerNorm, Conv2dLayerNorm)):
            return nn.GELU()
    return nn.GELU()



__all__ = [
    "_init_weights", "_check_norm_layer", "_check_activation",
    "conv1x1",
    "conv3x3",
    "Conv2dLayerNorm",
    "interpolate_pos_embed",
    "DepthSeparableConv2d",
    "ConvRefine",
    "LightConvRefine",
    "LighterConvRefine",
    "ConvDownsample",
    "LightConvDownsample",
    "LighterConvDownsample",
    "ConvUpsample",
    "LightConvUpsample",
    "LighterConvUpsample",
    "MultiScale",
    "ConvAdapter", "ViTAdapter",
]
