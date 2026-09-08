from torch import nn, Tensor
from torch.hub import load_state_dict_from_url
from typing import Optional

from .vgg import VGG
from .utils import make_vgg_layers, vgg_urls
from ..utils import _init_weights, ConvDownsample, _get_activation, _get_norm_layer

EPS = 1e-6


encoder_cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
decoder_cfg = [512, 512, 512, 256, 128]


class CSRNet(nn.Module):
    def __init__(
        self,
        model_name: str,
        block_size: Optional[int] = None,
        norm: str = "none",
        act: str = "none"
    ) -> None:
        super().__init__()
        assert model_name in ["vgg16", "vgg16_bn"], f"Model name should be one of ['vgg16', 'vgg16_bn'], but got {model_name}."
        assert block_size is None or block_size in [8, 16, 32], f"block_size should be one of [8, 16, 32], but got {block_size}."
        self.model_name = model_name

        vgg = VGG(make_vgg_layers(encoder_cfg, in_channels=3, batch_norm="bn" in model_name, dilation=1))
        vgg.load_state_dict(load_state_dict_from_url(vgg_urls[model_name]), strict=False)
        self.encoder = vgg.features
        self.encoder_reduction = 8
        self.encoder_channels = 512
        self.block_size = block_size if block_size is not None else 8

        if norm == "bn":
            norm_layer = nn.BatchNorm2d
        elif norm == "ln":
            norm_layer = nn.LayerNorm
        else:
            norm_layer = _get_norm_layer(vgg)
        
        if act == "relu":
            activation = nn.ReLU(inplace=True)
        elif act == "gelu":
            activation = nn.GELU()
        else:
            activation = _get_activation(vgg)

        if self.block_size == self.encoder_reduction:
            self.refiner = nn.Identity()
        elif self.block_size > self.encoder_reduction:
            if self.block_size == 32:
                self.refiner = nn.Sequential(
                    ConvDownsample(
                        in_channels=self.encoder_channels,
                        out_channels=self.encoder_channels,
                        norm_layer=norm_layer, 
                        activation=activation,
                    ),
                    ConvDownsample(
                        in_channels=self.encoder_channels,
                        out_channels=self.encoder_channels,
                        norm_layer=norm_layer, 
                        activation=activation,
                    )
                )
            elif self.block_size == 16:
                self.refiner = ConvDownsample(
                    in_channels=self.encoder_channels,
                    out_channels=self.encoder_channels,
                    norm_layer=norm_layer, 
                    activation=activation,
                )
        self.refiner_channels = self.encoder_channels
        self.refiner_reduction = self.block_size

        decoder = make_vgg_layers(decoder_cfg, in_channels=512, batch_norm="bn" in model_name, dilation=2)
        decoder.apply(_init_weights)
        self.decoder = decoder
        self.decoder_channels = decoder_cfg[-1]
        self.decoder_reduction = self.refiner_reduction

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


def _csrnet(block_size: Optional[int] = None, norm: str = "none", act: str = "none") -> CSRNet:
    return CSRNet("vgg16", block_size=block_size, norm=norm, act=act)

def _csrnet_bn(block_size: Optional[int] = None, norm: str = "none", act: str = "none") -> CSRNet:
    return CSRNet("vgg16_bn", block_size=block_size, norm=norm, act=act)
