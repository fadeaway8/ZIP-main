from torch import nn, Tensor
from torch.hub import load_state_dict_from_url
from typing import Optional

from .utils import make_vgg_layers, vgg_cfgs, vgg_urls
from ..utils import _init_weights, _get_norm_layer, _get_activation
from ..utils import  ConvDownsample, ConvUpsample


vgg_models = [
    "vgg11", "vgg11_bn",
    "vgg13", "vgg13_bn",
    "vgg16", "vgg16_bn",
    "vgg19", "vgg19_bn",
]

decoder_cfg = [512, 256, 128]


class VGGEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        block_size: Optional[int] = None,
        norm: str = "none",
        act: str = "none",
    ) -> None:
        super().__init__()
        assert model_name in vgg_models, f"Model name should be one of {vgg_models}, but got {model_name}."
        assert block_size is None or block_size in [8, 16, 32], f"Block size should be one of [8, 16, 32], but got {block_size}."
        self.model_name = model_name

        if model_name == "vgg11":
            self.encoder = vgg11()
        elif model_name == "vgg11_bn":
            self.encoder = vgg11_bn()
        elif model_name == "vgg13":
            self.encoder = vgg13()
        elif model_name == "vgg13_bn":
            self.encoder = vgg13_bn()
        elif model_name == "vgg16":
            self.encoder = vgg16()
        elif model_name == "vgg16_bn":
            self.encoder = vgg16_bn()
        elif model_name == "vgg19":
            self.encoder = vgg19()
        else:  # model_name == "vgg19_bn"
            self.encoder = vgg19_bn()
        
        self.encoder_channels = 512
        self.encoder_reduction = 16
        self.block_size = block_size if block_size is not None else self.encoder_reduction

        if norm == "bn":
            norm_layer = nn.BatchNorm2d
        elif norm == "ln":
            norm_layer = nn.LayerNorm
        else:
            norm_layer = _get_norm_layer(self.encoder)
        
        if act == "relu":
            activation = nn.ReLU(inplace=True)
        elif act == "gelu":
            activation = nn.GELU()
        else:
            activation = _get_activation(self.encoder)
        
        if self.encoder_reduction >= self.block_size:  # 8, 16
            self.refiner = ConvUpsample(
                in_channels=self.encoder_channels,
                out_channels=self.encoder_channels,
                scale_factor=self.encoder_reduction // self.block_size,
                norm_layer=norm_layer,
                activation=activation,
            )
        else: # 32
            self.refiner = ConvDownsample(
                in_channels=self.encoder_channels,
                out_channels=self.encoder_channels,
                norm_layer=norm_layer,
                activation=activation,
            )
        self.refiner_channels = self.encoder_channels
        self.refiner_reduction = self.block_size

        self.decoder = nn.Identity()
        self.decoder_channels = self.encoder_channels
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


class VGGEncoderDecoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        block_size: Optional[int] = None,
        norm: str = "none",
        act: str = "none",
    ) -> None:
        super().__init__()
        assert model_name in vgg_models, f"Model name should be one of {vgg_models}, but got {model_name}."
        assert block_size is None or block_size in [8, 16, 32], f"Block size should be one of [8, 16, 32], but got {block_size}."
        self.model_name = model_name

        if model_name == "vgg11":
            encoder = vgg11()
        elif model_name == "vgg11_bn":
            encoder = vgg11_bn()
        elif model_name == "vgg13":
            encoder = vgg13()
        elif model_name == "vgg13_bn":
            encoder = vgg13_bn()
        elif model_name == "vgg16":
            encoder = vgg16()
        elif model_name == "vgg16_bn":
            encoder = vgg16_bn()
        elif model_name == "vgg19":
            encoder = vgg19()
        else:  # model_name == "vgg19_bn"
            encoder = vgg19_bn()
        
        encoder_channels = 512
        encoder_reduction = 16
        decoder = make_vgg_layers(decoder_cfg, in_channels=encoder_channels, batch_norm="bn" in model_name, dilation=1)
        decoder.apply(_init_weights)

        if norm == "bn":
            norm_layer = nn.BatchNorm2d
        elif norm == "ln":
            norm_layer = nn.LayerNorm
        else:
            norm_layer = _get_norm_layer(encoder)
        
        if act == "relu":
            activation = nn.ReLU(inplace=True)
        elif act == "gelu":
            activation = nn.GELU()
        else:
            activation = _get_activation(encoder)

        self.encoder = nn.Sequential(encoder, decoder)
        self.encoder_channels = decoder_cfg[-1]
        self.encoder_reduction = encoder_reduction
        self.block_size = block_size if block_size is not None else self.encoder_reduction
        
        if self.encoder_reduction >= self.block_size:
            self.refiner = ConvUpsample(
                in_channels=self.encoder_channels,
                out_channels=self.encoder_channels,
                scale_factor=self.encoder_reduction // self.block_size,
                norm_layer=norm_layer,
                activation=activation,
            )
        else:
            self.refiner = ConvDownsample(
                in_channels=self.encoder_channels,
                out_channels=self.encoder_channels,
                norm_layer=norm_layer,
                activation=activation,
            )
        self.refiner_channels = self.encoder_channels
        self.refiner_reduction = self.block_size

        self.decoder = nn.Identity()
        self.decoder_channels = self.refiner_channels
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


class VGG(nn.Module):
    def __init__(
        self,
        features: nn.Module,
    ) -> None:
        super().__init__()
        self.features = features

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return x


def vgg11() -> VGG:
    model = VGG(make_vgg_layers(vgg_cfgs["A"]))
    model.load_state_dict(state_dict=load_state_dict_from_url(vgg_urls["vgg11"]), strict=False)
    return model

def vgg11_bn() -> VGG:
    model = VGG(make_vgg_layers(vgg_cfgs["A"], batch_norm=True))
    model.load_state_dict(state_dict=load_state_dict_from_url(vgg_urls["vgg11_bn"]), strict=False)
    return model

def vgg13() -> VGG:
    model = VGG(make_vgg_layers(vgg_cfgs["B"]))
    model.load_state_dict(state_dict=load_state_dict_from_url(vgg_urls["vgg13"]), strict=False)
    return model

def vgg13_bn() -> VGG:
    model = VGG(make_vgg_layers(vgg_cfgs["B"], batch_norm=True))
    model.load_state_dict(state_dict=load_state_dict_from_url(vgg_urls["vgg13_bn"]), strict=False)
    return model

def vgg16() -> VGG:
    model = VGG(make_vgg_layers(vgg_cfgs["D"]))
    model.load_state_dict(state_dict=load_state_dict_from_url(vgg_urls["vgg16"]), strict=False)
    return model

def vgg16_bn() -> VGG:
    model = VGG(make_vgg_layers(vgg_cfgs["D"], batch_norm=True))
    model.load_state_dict(state_dict=load_state_dict_from_url(vgg_urls["vgg16_bn"]), strict=False)
    return model

def vgg19() -> VGG:
    model = VGG(make_vgg_layers(vgg_cfgs["E"]))
    model.load_state_dict(state_dict=load_state_dict_from_url(vgg_urls["vgg19"]), strict=False)
    return model

def vgg19_bn() -> VGG:
    model = VGG(make_vgg_layers(vgg_cfgs["E"], batch_norm=True))
    model.load_state_dict(state_dict=load_state_dict_from_url(vgg_urls["vgg19_bn"]), strict=False)
    return model

def _vgg_encoder(model_name: str, block_size: Optional[int] = None, norm: str = "none", act: str = "none") -> VGGEncoder:
    return VGGEncoder(model_name, block_size, norm=norm, act=act)

def _vgg_encoder_decoder(model_name: str, block_size: Optional[int] = None, norm: str = "none", act: str = "none") -> VGGEncoderDecoder:
    return VGGEncoderDecoder(model_name, block_size, norm=norm, act=act)
