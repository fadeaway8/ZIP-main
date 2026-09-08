import torch
from torch import nn, Tensor
import timm
from einops import rearrange
import torch.nn.functional as F

import math
from typing import Optional, Tuple
from ..utils import ConvUpsample, ConvDownsample, _get_activation, _get_norm_layer, ConvRefine


supported_vit_backbones = [
    # Tiny
    "vit_tiny_patch16_224", "vit_tiny_patch16_384",
    # Small
    "vit_small_patch8_224",
    "vit_small_patch16_224", "vit_small_patch16_384",
    "vit_small_patch32_224", "vit_small_patch32_384",
    # Base
    "vit_base_patch8_224", 
    "vit_base_patch16_224", "vit_base_patch16_384",
    "vit_base_patch32_224", "vit_base_patch32_384",
    # Large
    "vit_large_patch16_224", "vit_large_patch16_384", 
    "vit_large_patch32_224", "vit_large_patch32_384",
    # Huge
    "vit_huge_patch14_224",
]


refiner_channels = {
    "vit_tiny_patch16_224": 192,
    "vit_tiny_patch16_384": 192,
    "vit_small_patch8_224": 384,
    "vit_small_patch16_224": 384,
    "vit_small_patch16_384": 384,
    "vit_small_patch32_224": 384,
    "vit_small_patch32_384": 384,
    "vit_base_patch8_224": 768,
    "vit_base_patch16_224": 768,
    "vit_base_patch16_384": 768,
    "vit_base_patch32_224": 768,
    "vit_base_patch32_384": 768,
    "vit_large_patch16_224": 1024,
    "vit_large_patch16_384": 1024,
    "vit_large_patch32_224": 1024,
    "vit_large_patch32_384": 1024,
}

refiner_groups = {
    "vit_tiny_patch16_224": 1,
    "vit_tiny_patch16_384": 1,
    "vit_small_patch8_224": 1,
    "vit_small_patch16_224": 1,
    "vit_small_patch16_384": 1,
    "vit_small_patch32_224": 1,
    "vit_small_patch32_384": 1,
    "vit_base_patch8_224": 1,
    "vit_base_patch16_224": 1,
    "vit_base_patch16_384": 1,
    "vit_base_patch32_224": 1,
    "vit_base_patch32_384": 1,
    "vit_large_patch16_224": 1,
    "vit_large_patch16_384": 1,
    "vit_large_patch32_224": 1,
    "vit_large_patch32_384": 1,
}


class ViT(nn.Module):
    def __init__(
        self,
        model_name: str,
        block_size: Optional[int] = None,
        num_vpt: int = 32,
        vpt_drop: float = 0.0,
        input_size: Optional[Tuple[int, int]] = None,
        norm: str = "none",
        act: str = "none"
    ) -> None:
        super().__init__()
        assert model_name in supported_vit_backbones, f"Model {model_name} not supported"
        assert num_vpt >= 0, f"Number of VPT tokens should be greater than 0, but got {num_vpt}."
        self.model_name = model_name

        self.num_vpt = num_vpt
        self.vpt_drop = vpt_drop

        model = timm.create_model(model_name, pretrained=True)

        self.input_size = input_size if input_size is not None else model.patch_embed.img_size
        self.pretrain_size = model.patch_embed.img_size
        self.patch_size = model.patch_embed.patch_size

        if self.patch_size[0] in [8, 16, 32]:
            assert block_size is None or block_size in [8, 16, 32], f"Block size should be one of [8, 16, 32], but got {block_size}."
        else:  # patch_size == 14
            assert block_size is None or block_size in [7, 14, 28], f"Block size should be one of [7, 14, 28], but got {block_size}."

        self.num_layers = len(model.blocks)
        self.embed_dim = model.cls_token.shape[-1]

        if self.num_vpt > 0:  # Use visual prompt tuning so freeze the backbone
            for param in model.parameters():
                param.requires_grad = False
            
            # Setup VPT tokens
            val = math.sqrt(6. / float(3 * self.patch_size[0] + self.embed_dim))
            for idx in range(self.num_layers):
                setattr(self, f"vpt_{idx}", nn.Parameter(torch.empty(self.num_vpt, self.embed_dim)))
                nn.init.uniform_(getattr(self, f"vpt_{idx}"), -val, val)
                setattr(self, f"vpt_drop_{idx}", nn.Dropout(self.vpt_drop))
        
        self.patch_embed = model.patch_embed
        self.cls_token = model.cls_token
        self.pos_embed = model.pos_embed
        self.pos_drop = model.pos_drop
        self.patch_drop = model.patch_drop
        self.norm_pre = model.norm_pre

        self.blocks = model.blocks
        self.norm = model.norm

        self.encoder_channels = self.embed_dim
        self.encoder_reduction = self.patch_size[0]
        self.block_size = block_size if block_size is not None else self.encoder_reduction

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

        if self.block_size < self.encoder_reduction:
            assert self.block_size == self.encoder_reduction // 2, f"Block size should be half of the encoder reduction, but got {self.block_size} and {self.encoder_reduction}."
            self.refiner = ConvUpsample(
                in_channels=self.encoder_channels,
                out_channels=self.encoder_channels,
                norm_layer=norm_layer,
                activation=activation,
            )
        elif self.block_size > self.encoder_reduction:
            assert self.block_size == self.encoder_reduction * 2, f"Block size should be double of the encoder reduction, but got {self.block_size} and {self.encoder_reduction}."
            self.refiner = ConvDownsample(
                in_channels=self.encoder_channels,
                out_channels=self.encoder_channels,
                norm_layer=norm_layer,
                activation=activation,
            )
        else:
            self.refiner = ConvRefine(
                in_channels=self.encoder_channels,
                out_channels=self.encoder_channels,
                norm_layer=norm_layer,
                activation=activation,
            )

        self.refiner_channels = self.encoder_channels
        self.refiner_reduction = self.block_size

        self.decoder = nn.Identity()
        self.decoder_channels = self.refiner_channels
        self.reduction = self.refiner_reduction

        # Adjust the positional embedding to match the new input size
        self._adjust_pos_embed()
    
    def _adjust_pos_embed(self) -> Tensor:
        """
        Adjust the positional embedding to match the spatial resolution of the feature map.

        Args:
            orig_h, orig_w: The original spatial resolution of the image.
            new_h, new_w: The new spatial resolution of the image.
        """
        self.pos_embed = nn.Parameter(self._interpolate_pos_embed(self.pretrain_size[0], self.pretrain_size[1], self.input_size[0], self.input_size[1]), requires_grad=self.num_vpt == 0)

    def _interpolate_pos_embed(self, orig_h: int, orig_w: int, new_h: int, new_w: int) -> Tensor:
        """
        Interpolate the positional embedding to match the spatial resolution of the feature map.

        Args:
            orig_h, orig_w: The original spatial resolution of the image.
            new_h, new_w: The new spatial resolution of the image.
        """
        if (orig_h, orig_w) == (new_h, new_w):
            return self.pos_embed  # (1, (h * w + 1), d)
        
        orig_h_patches, orig_w_patches = orig_h // self.patch_size[0], orig_w // self.patch_size[1]
        new_h_patches, new_w_patches = new_h // self.patch_size[0], new_w // self.patch_size[1]
        class_pos_embed, patch_pos_embed = self.pos_embed[:, :1, :], self.pos_embed[:, 1:, :]
        patch_pos_embed = rearrange(patch_pos_embed, "1 (h w) d -> 1 d h w", h=orig_h_patches, w=orig_w_patches)
        patch_pos_embed = F.interpolate(patch_pos_embed, size=(new_h_patches, new_w_patches), mode="bicubic", antialias=True)
        patch_pos_embed = rearrange(patch_pos_embed, "1 d h w -> 1 (h w) d")
        pos_embed = torch.cat((class_pos_embed, patch_pos_embed), dim=1)
        return pos_embed

    def train(self, mode: bool = True):
        if self.num_vpt > 0 and mode:
            self.patch_embed.eval()
            self.pos_drop.eval()
            self.patch_drop.eval()
            self.norm_pre.eval()

            self.blocks.eval()
            self.norm.eval()
        
            for idx in range(self.num_layers):
                getattr(self, f"vpt_drop_{idx}").train()
            
            self.refiner.train()
            self.decoder.train()

        else:
            for module in self.children():
                module.train(mode)

    def _prepare_vpt(self, layer: int, batch_size: int, device: torch.device) -> Tensor:
        vpt = getattr(self, f"vpt_{layer}").unsqueeze(0).expand(batch_size, -1, -1).to(device)  # (batch_size, num_vpt, embed_dim)
        vpt = getattr(self, f"vpt_drop_{layer}")(vpt)

        return vpt

    def _forward_patch_embed(self, x: Tensor) -> Tensor:
        # This step performs 1) embed x into patches; 2) append the class token; 3) add positional embeddings.
        assert len(x.shape) == 4, f"Expected input to have shape (batch_size, 3, height, width), but got {x.shape}"
        batch_size, _, height, width = x.shape

        # Step 1: Embed x into patches
        x = self.patch_embed(x)  # (b, h * w, d)

        # Step 2: Append the class token
        cls_token = self.cls_token.expand(batch_size, 1, -1)
        x = torch.cat([cls_token, x], dim=1)

        # Step 3: Add positional embeddings
        pos_embed = self._interpolate_pos_embed(orig_h=self.input_size[0], orig_w=self.input_size[1], new_h=height, new_w=width).expand(batch_size, -1, -1)
        x = self.pos_drop(x + pos_embed)
        return x

    def _forward_vpt(self, x: Tensor, idx: int) -> Tensor:
        batch_size = x.shape[0]
        device = x.device

        # Assemble
        vpt = self._prepare_vpt(idx, batch_size, device)
        x = torch.cat([
            x[:, :1, :],  # class token
            vpt,
            x[:, 1:, :]  # patches
        ], dim=1)

        # Forward
        x = self.blocks[idx](x)

        # Disassemble
        x = torch.cat([
            x[:, :1, :],  # class token
            x[:, 1 + self.num_vpt:, :]  # patches
        ], dim=1)

        return x

    def _forward(self, x: Tensor, idx: int) -> Tensor:
        x = self.blocks[idx](x)
        return x

    def encode(self, x: Tensor) -> Tensor:
        orig_h, orig_w = x.shape[-2:]
        num_patches_h, num_patches_w = orig_h // self.patch_size[0], orig_w // self.patch_size[1]

        x = self._forward_patch_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        for idx in range(self.num_layers):
            x = self._forward_vpt(x, idx) if self.num_vpt > 0 else self._forward(x, idx)
        
        x = self.norm(x)
        x = x[:, 1:, :]
        x = rearrange(x, "b (h w) d -> b d h w", h=num_patches_h, w=num_patches_w)
        return x

    def refine(self, x: Tensor) -> Tensor:
        return self.refiner(x)
    
    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encode(x)
        x = self.refine(x)
        x = self.decode(x)
        return x


def _vit(
    model_name: str,
    block_size: Optional[int] = None,
    num_vpt: int = 32,
    vpt_drop: float = 0.0,
    input_size: Optional[Tuple[int, int]] = None,
    norm: str = "none",
    act: str = "none"
) -> ViT:
    model = ViT(
        model_name=model_name,
        block_size=block_size,
        num_vpt=num_vpt,
        vpt_drop=vpt_drop,
        input_size=input_size,
        norm=norm,
        act=act
    )
    return model
