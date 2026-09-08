import timm
import torch.nn.functional as F
from torch import nn, Tensor
from functools import partial
from typing import Optional

from ..utils import ConvRefine, _get_norm_layer, _get_activation


available_hrnets = [
    "hrnet_w18", "hrnet_w18_small", "hrnet_w18_small_v2",
    "hrnet_w30", "hrnet_w32", "hrnet_w40", "hrnet_w44", "hrnet_w48", "hrnet_w64",
]


class HRNet(nn.Module):
    def __init__(
        self,
        model_name: str,
        block_size: Optional[int] = None,
        norm: str = "none",
        act: str = "none"
    ) -> None:
        super().__init__()
        assert model_name in available_hrnets, f"Model name should be one of {available_hrnets}"
        assert block_size is None or block_size in [8, 16, 32], f"block_size should be one of [8, 16, 32], but got {block_size}."
        self.model_name = model_name
        self.block_size = block_size if block_size is not None else 32

        model = timm.create_model(model_name, pretrained=True)

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.act1 = model.act1
        self.conv2 = model.conv2
        self.bn2 = model.bn2
        self.act2 = model.act2

        self.layer1 = model.layer1

        self.transition1 = model.transition1
        self.stage2 = model.stage2

        self.transition2 = model.transition2
        self.stage3 = model.stage3

        self.transition3 = model.transition3
        self.stage4 = model.stage4

        incre_modules = model.incre_modules
        downsamp_modules = model.downsamp_modules

        assert len(incre_modules) == 4, f"Expected 4 incre_modules, got {len(self.incre_modules)}"
        assert len(downsamp_modules) == 3, f"Expected 3 downsamp_modules, got {len(self.downsamp_modules)}"

        self.out_channels_4 = incre_modules[0][0].downsample[0].out_channels
        self.out_channels_8 = incre_modules[1][0].downsample[0].out_channels
        self.out_channels_16 = incre_modules[2][0].downsample[0].out_channels
        self.out_channels_32 = incre_modules[3][0].downsample[0].out_channels

        if self.block_size == 8:
            self.encoder_reduction = 8
            self.encoder_channels = self.out_channels_8
            self.incre_modules = incre_modules[:2]
            self.downsamp_modules = downsamp_modules[:1]

            self.refiner = nn.Identity()
            self.refiner_reduction = 8
            self.refiner_channels = self.out_channels_8
        
        elif self.block_size == 16:
            self.encoder_reduction = 16
            self.encoder_channels = self.out_channels_16
            self.incre_modules = incre_modules[:3]
            self.downsamp_modules = downsamp_modules[:2]

            self.refiner = nn.Identity()
            self.refiner_reduction = 16
            self.refiner_channels = self.out_channels_16

        else:  # self.block_size == 32
            self.encoder_reduction = 32
            self.encoder_channels = self.out_channels_32
            self.incre_modules = incre_modules
            self.downsamp_modules = downsamp_modules

            self.refiner = nn.Identity()
            self.refiner_reduction = 32
            self.refiner_channels = self.out_channels_32

        # define the decoder
        if self.refiner_channels <= 512:
            groups = 1
        elif self.refiner_channels <= 1024:
            groups = 2
        elif self.refiner_channels <= 2048:
            groups = 4
        else:
            groups = 8
        
        if norm == "bn":
            norm_layer = nn.BatchNorm2d
        elif norm == "ln":
            norm_layer = nn.LayerNorm
        else:
            norm_layer = _get_norm_layer(model)

        if act == "relu":
            activation = nn.ReLU(inplace=True)
        elif act == "gelu":
            activation = nn.GELU()
        else:
            activation = _get_activation(model)
        
        decoder_block = partial(ConvRefine, groups=groups, norm_layer=norm_layer, activation=activation)
        if self.refiner_channels <= 256:
            self.decoder = nn.Identity()
            self.decoder_channels = self.refiner_channels
        elif self.refiner_channels <= 512:
            self.decoder = decoder_block(self.refiner_channels, self.refiner_channels // 2)
            self.decoder_channels = self.refiner_channels // 2
        elif self.refiner_channels <= 1024:
            self.decoder = nn.Sequential(
                decoder_block(self.refiner_channels, self.refiner_channels // 2),
                decoder_block(self.refiner_channels // 2, self.refiner_channels // 4),
            )
            self.decoder_channels = self.refiner_channels // 4
        else:
            self.decoder = nn.Sequential(
                decoder_block(self.refiner_channels, self.refiner_channels // 2),
                decoder_block(self.refiner_channels // 2, self.refiner_channels // 4),
                decoder_block(self.refiner_channels // 4, self.refiner_channels // 8),
            )
            self.decoder_channels = self.refiner_channels // 8
        
        self.decoder_reduction = self.refiner_reduction

    def _interpolate(self, x: Tensor) -> Tensor:
        # This method adjust the spatial dimensions of the input tensor so that it can be divided by 32.
        if x.shape[-1] % 32 != 0 or x.shape[-2] % 32 != 0:
            new_h = int(round(x.shape[-2] / 32) * 32)
            new_w = int(round(x.shape[-1] / 32) * 32)
            return F.interpolate(x, size=(new_h, new_w), mode="bicubic", align_corners=False)
        
        return x


    def encode(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.layer1(x)

        x = [t(x) for t in self.transition1]
        x = self.stage2(x)

        x = [t(x[-1]) if not isinstance(t, nn.Identity) else x[i] for i, t in enumerate(self.transition2)]
        x = self.stage3(x)

        x = [t(x[-1]) if not isinstance(t, nn.Identity) else x[i] for i, t in enumerate(self.transition3)]
        x = self.stage4(x)

        assert len(x) == 4, f"Expected 4 outputs, got {len(x)}"

        feats = None
        for i, incre in enumerate(self.incre_modules):
            if feats is None:
                feats = incre(x[i])
            else:
                down = self.downsamp_modules[i - 1]  # needed for torchscript module indexing
                feats = incre(x[i]) + down.forward(feats)
        
        return feats

    def refine(self, x: Tensor) -> Tensor:
        return self.refiner(x)
    
    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self._interpolate(x)
        x = self.encode(x)
        x = self.refine(x)
        x = self.decode(x)
        return x


def _hrnet(model_name: str, block_size: Optional[int] = None, norm: str = "none", act: str = "none") -> HRNet:
    return HRNet(model_name, block_size, norm, act)