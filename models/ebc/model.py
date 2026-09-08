import torch
from torch import nn, Tensor
from einops import rearrange

from typing import Tuple, Union, Dict, Optional, List
from functools import partial

from .cannet import _cannet, _cannet_bn
from .csrnet import _csrnet, _csrnet_bn
from .vgg import _vgg_encoder_decoder, _vgg_encoder
from .vit import _vit, supported_vit_backbones
from .timm_models import _timm_model
from .timm_models import regular_models as timm_regular_models, heavy_models as timm_heavy_models, light_models as timm_light_models, lighter_models as timm_lighter_models
from .hrnet import _hrnet, available_hrnets

from ..utils import conv1x1


regular_models = [
    "csrnet", "csrnet_bn",
    "cannet", "cannet_bn",
    "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn",
    "vgg11_ae", "vgg11_bn_ae", "vgg13_ae", "vgg13_bn_ae", "vgg16_ae", "vgg16_bn_ae", "vgg19_ae", "vgg19_bn_ae",
    *timm_regular_models,
    *available_hrnets,
]

heavy_models = timm_heavy_models

light_models = timm_light_models

lighter_models = timm_lighter_models

transformer_models = supported_vit_backbones

supported_models = regular_models + heavy_models + light_models + lighter_models + transformer_models



class EBC(nn.Module):
    def __init__(
        self,
        model_name: str,
        block_size: int,
        bins: List[Tuple[float, float]],
        bin_centers: List[float],
        zero_inflated: bool = True,
        num_vpt: Optional[int] = None,
        vpt_drop: Optional[float] = None,
        input_size: Optional[int] = None,
        norm: str = "none",
        act: str = "none"
    ) -> None:
        super().__init__()
        assert model_name in supported_models, f"Model name should be one of {supported_models}, but got {model_name}."
        self.model_name = model_name

        if input_size is not None:
            input_size = (input_size, input_size) if isinstance(input_size, int) else input_size
            assert len(input_size) == 2 and input_size[0] > 0 and input_size[1] > 0, f"Expected input_size to be a tuple of two positive integers, got {input_size}"
        self.input_size = input_size

        assert len(bins) == len(bin_centers), f"Expected bins and bin_centers to have the same length, got {len(bins)} and {len(bin_centers)}"
        assert len(bins) >= 2, f"Expected at least 2 bins, got {len(bins)}"
        assert all(len(b) == 2 for b in bins), f"Expected bins to be a list of tuples of length 2, got {bins}"
        bins = [(float(b[0]), float(b[1])) for b in bins]
        assert all(bin[0] <= p <= bin[1] for bin, p in zip(bins, bin_centers)), f"Expected bin_centers to be within the range of the corresponding bin, got {bins} and {bin_centers}"

        self.block_size = block_size
        self.bins = bins
        self.register_buffer("bin_centers", torch.tensor(bin_centers, dtype=torch.float32, requires_grad=False).view(1, -1, 1, 1))

        self.zero_inflated = zero_inflated
        self.num_vpt = num_vpt
        self.vpt_drop = vpt_drop
        self.input_size = input_size

        self.norm = norm
        self.act = act

        self._build_backbone()
        self._build_head()
    
    def _build_backbone(self) -> None:
        model_name = self.model_name
        if model_name == "csrnet":
            self.backbone = _csrnet(self.block_size, self.norm, self.act)
        elif model_name == "csrnet_bn":
            self.backbone = _csrnet_bn(self.block_size, self.norm, self.act)
        elif model_name == "cannet":
            self.backbone = _cannet(self.block_size, self.norm, self.act)
        elif model_name == "cannet_bn":
            self.backbone = _cannet_bn(self.block_size, self.norm, self.act)
        elif model_name == "vgg11":
            self.backbone = _vgg_encoder("vgg11", self.block_size, self.norm, self.act)
        elif model_name == "vgg11_ae":
            self.backbone = _vgg_encoder_decoder("vgg11", self.block_size, self.norm, self.act)
        elif model_name == "vgg11_bn":
            self.backbone = _vgg_encoder("vgg11_bn", self.block_size, self.norm, self.act)
        elif model_name == "vgg11_bn_ae":
            self.backbone = _vgg_encoder_decoder("vgg11_bn", self.block_size, self.norm, self.act) 
        elif model_name == "vgg13":
            self.backbone = _vgg_encoder("vgg13", self.block_size, self.norm, self.act)
        elif model_name == "vgg13_ae":
            self.backbone = _vgg_encoder_decoder("vgg13", self.block_size, self.norm, self.act)
        elif model_name == "vgg13_bn":
            self.backbone = _vgg_encoder("vgg13_bn", self.block_size, self.norm, self.act)
        elif model_name == "vgg13_bn_ae":
            self.backbone = _vgg_encoder_decoder("vgg13_bn", self.block_size, self.norm, self.act)
        elif model_name == "vgg16":
            self.backbone = _vgg_encoder("vgg16", self.block_size, self.norm, self.act)
        elif model_name == "vgg16_ae":
            self.backbone = _vgg_encoder_decoder("vgg16", self.block_size, self.norm, self.act)
        elif model_name == "vgg16_bn":
            self.backbone = _vgg_encoder("vgg16_bn", self.block_size, self.norm, self.act)
        elif model_name == "vgg16_bn_ae":
            self.backbone = _vgg_encoder_decoder("vgg16_bn", self.block_size, self.norm, self.act)
        elif model_name == "vgg19":
            self.backbone = _vgg_encoder("vgg19", self.block_size, self.norm, self.act)
        elif model_name == "vgg19_ae":
            self.backbone = _vgg_encoder_decoder("vgg19", self.block_size, self.norm, self.act)
        elif model_name == "vgg19_bn":
            self.backbone = _vgg_encoder("vgg19_bn", self.block_size, self.norm, self.act)
        elif model_name == "vgg19_bn_ae":
            self.backbone = _vgg_encoder_decoder("vgg19_bn", self.block_size, self.norm, self.act)
        elif model_name in supported_vit_backbones:
            self.backbone = _vit(model_name, block_size=self.block_size, num_vpt=self.num_vpt, vpt_drop=self.vpt_drop, input_size=self.input_size, norm=self.norm, act=self.act)
        elif model_name in available_hrnets:
            self.backbone = _hrnet(model_name, block_size=self.block_size, norm=self.norm, act=self.act)
        else:
            self.backbone = _timm_model(model_name, self.block_size, self.norm, self.act)

    def _build_head(self) -> None:
        channels = self.backbone.decoder_channels
        if self.zero_inflated:
            self.bin_head = conv1x1(
                in_channels=channels,
                out_channels=len(self.bins) - 1,
            )
            self.pi_head = conv1x1(
                in_channels=channels,
                out_channels=2,
            )  # this models structural 0s.
        else:
            self.bin_head = conv1x1(
                in_channels=channels,
                out_channels=len(self.bins),
            )

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x = self.backbone(x)

        if self.zero_inflated:
            logit_pi_maps = self.pi_head(x)  # shape: (B, 2, H, W)
            logit_maps = self.bin_head(x)  # shape: (B, C, H, W)
            lambda_maps = (logit_maps.softmax(dim=1) * self.bin_centers[:, 1:]).sum(dim=1, keepdim=True)  # shape: (B, 1, H, W)

            # logit_pi_maps.softmax(dim=1)[:, 0] is the probability of zeros
            den_maps = logit_pi_maps.softmax(dim=1)[:, 1:] * lambda_maps  # expectation of the Poisson distribution

            if self.training:
                return logit_pi_maps, logit_maps, lambda_maps, den_maps
            else:
                return den_maps
            
        else:
            logit_maps = self.bin_head(x)
            den_maps = (logit_maps.softmax(dim=1) * self.bin_centers).sum(dim=1, keepdim=True)

            if self.training:
                return logit_maps, den_maps
            else:
                return den_maps


def _ebc(
    model_name: str,
    block_size: int,
    bins: List[Tuple[float, float]],
    bin_centers: List[float],
    zero_inflated: bool = True,
    num_vpt: Optional[int] = None,
    vpt_drop: Optional[float] = None,
    input_size: Optional[int] = None,
    norm: str = "none",
    act: str = "none"
) -> EBC:
    return EBC(
        model_name=model_name,
        block_size=block_size,
        bins=bins,
        bin_centers=bin_centers,
        zero_inflated=zero_inflated,
        num_vpt=num_vpt,
        vpt_drop=vpt_drop,
        input_size=input_size,
        norm=norm,
        act=act
    )
