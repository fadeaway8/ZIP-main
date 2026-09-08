import torch
from torch import nn, Tensor
import math
from einops import rearrange
import open_clip
from peft import get_peft_model, LoraConfig
from typing import Optional, Tuple

from ..utils import interpolate_pos_embed, ViTAdapter
# from ..utils import TransformerRefine, TransformerDownsample, TransformerUpsample
from ..utils import ConvRefine, ConvDownsample, ConvUpsample
from ..utils import _get_norm_layer, _get_activation


vit_names_and_weights = {
    "ViT-B-32": [
        "openai",
        "laion400m_e31", "laion400m_e32", "laion2b_e16", "laion2b_s34b_b79k",
        "datacomp_xl_s13b_b90k", "datacomp_m_s128m_b4k", "datacomp_s_s13m_b4k",
        "commonpool_m_clip_s128m_b4k", "commonpool_m_laion_s128m_b4k", "commonpool_m_image_s128m_b4k", "commonpool_m_text_s128m_b4k", "commonpool_m_basic_s128m_b4k", "commonpool_m_s128m_b4k",
        "commonpool_s_clip_s13m_b4k", "commonpool_s_laion_s13m_b4k", "commonpool_s_image_s13m_b4k", "commonpool_s_text_s13m_b4k", "commonpool_s_basic_s13m_b4k", "commonpool_s_s13m_b4k",
    ],
    "ViT-B_32-256": ["datacomp_s34b_b86k"],
    "ViT-B-16": [
        "openai",
        "laion400m_e31", "laion400m_e32", "laion2b_s34b_b88k",
        "datacomp_xl_s13b_b90k", "datacomp_l_s1b_b8k",
        "commonpool_l_clip_s1b_b8k", "commonpool_l_laion_s1b_b8k", "commonpool_l_image_s1b_b8k", "commonpool_l_text_s1b_b8k", "commonpool_l_basic_s1b_b8k", "commonpool_l_s1b_b8k",
        "dfn2b"
    ],
    "ViT-L-14": [
        "openai",
        "laion400m_e31", "laion400m_e32", "laion2b_s32b_b82k",
        "datacomp_xl_s13b_b90k",
        "commonpool_xl_clip_s13b_b90k", "commonpool_xl_laion_s13b_b90k", "commonpool_xl_s13b_b90k"
    ],
    "ViT-L-14-336": ["openai"],
    "ViT-H-14": ["laion2b_s32b_b79k"],
    "ViT-g-14": ["laion2b_s12b_b42k", "laion2b_s34b_b88k"],
    "ViT-bigG-14": ["laion2b_s39b_b160k"],
}


refiner_channels = {
    "ViT-B-32": 768,
    "ViT-B-32-256": 768,
    "ViT-B-16": 768,
    "ViT-L-14": 1024,
    "ViT-L-14-336": 1024,
    "ViT-H-14": 1280,
    "ViT-g-14": 1408,
    "ViT-bigG-14": 1664,
}

refiner_groups = {
    "ViT-B-32": 1,
    "ViT-B-32-256": 1,
    "ViT-B-16": 1,
    "ViT-L-14": 1,
    "ViT-L-14-336": 1,
    "ViT-H-14": 1,
    "ViT-g-14": refiner_channels["ViT-g-14"] // 704,  # 2
    "ViT-bigG-14": refiner_channels["ViT-bigG-14"] // 416,  # 4
}



class ViT(nn.Module):
    def __init__(
        self,
        model_name: str,
        weight_name: str,
        block_size: int = 16,
        num_vpt: int = 32,
        vpt_drop: float = 0.0,
        adapter: bool = False,
        adapter_reduction: int = 4,
        input_size: Optional[Tuple[int, int]] = None,
        norm: str = "none",
        act: str = "none"
    ) -> None:
        super(ViT, self).__init__()
        assert model_name in vit_names_and_weights, f"Model name should be one of {list(vit_names_and_weights.keys())}, but got {model_name}."
        assert weight_name in vit_names_and_weights[model_name], f"Pretrained should be one of {vit_names_and_weights[model_name]}, but got {weight_name}."
        if adapter:
            assert num_vpt is None or num_vpt == 0, "num_vpt should be None or 0 when using adapter."
            assert vpt_drop is None or vpt_drop == 0.0, "vpt_drop should be None or 0.0 when using adapter."
        else:
            assert num_vpt > 0, f"Number of VPT tokens should be greater than 0, but got {num_vpt}."
            assert 0.0 <= vpt_drop < 1.0, f"VPT dropout should be in [0.0, 1.0), but got {vpt_drop}."

        self.model_name, self.weight_name = model_name, weight_name
        self.block_size = block_size
        self.num_vpt = num_vpt
        self.vpt_drop = vpt_drop
        self.adapter = adapter

        model = open_clip.create_model_from_pretrained(model_name, weight_name, return_transform=False).visual

        # Always freeze the parameters of the model
        for param in model.parameters():
            param.requires_grad = False

        # Setup the model
        self.input_size = input_size if input_size is not None else model.image_size
        self.pretrain_size = model.image_size
        self.patch_size = model.patch_size
        self.class_embedding = model.class_embedding
        self.positional_embedding = model.positional_embedding
        self.embed_dim = model.class_embedding.shape[-1]

        self.conv1 = model.conv1
        self.ln_pre = model.ln_pre
        self.resblocks = model.transformer.resblocks
        self.num_layers = len(self.resblocks)
        self.ln_post = model.ln_post

        # Setup VPT tokens
        val = math.sqrt(6. / float(3 * self.patch_size[0] + self.embed_dim))
        for idx in range(self.num_layers):
            if self.adapter:
                setattr(self, f"adapter{idx}", ViTAdapter(
                    in_channels=self.embed_dim,
                    bottleneck_channels=self.embed_dim // adapter_reduction,
                ))
            else:
                setattr(self, f"vpt_{idx}", nn.Parameter(torch.empty(self.num_vpt, self.embed_dim)))
                nn.init.uniform_(getattr(self, f"vpt_{idx}"), -val, val)
                setattr(self, f"vpt_drop_{idx}", nn.Dropout(self.vpt_drop))
        
        # Adjust the positional embedding to match the new input size
        self._adjust_pos_embed()

        in_features, out_features = model.proj.shape
        self.in_features = in_features
        self.out_features = out_features

        patch_size = self.patch_size[0]
        if patch_size in [16, 32]:
            assert block_size in [8, 16, 32], f"Patch size is 32, but got block size {block_size}."
        else:  # patch_size == 14
            assert block_size in [7, 14, 28], f"Patch size is 14, but got block size {block_size}."

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
        
        if block_size == patch_size:
            self.refiner = ConvRefine(
                in_channels=self.in_features,
                out_channels=self.in_features,
                norm_layer=norm_layer,
                activation=activation,
                groups=refiner_groups[self.model_name],
            )
        
        elif block_size < patch_size:  # upsample
            if block_size == 8 and patch_size == 32:
                self.refiner = nn.Sequential(
                    ConvUpsample(
                        in_channels=self.in_features,
                        out_channels=self.in_features,
                        norm_layer=norm_layer,
                        activation=activation,
                        groups=refiner_groups[self.model_name],
                    ),
                    ConvUpsample(
                        in_channels=self.in_features,
                        out_channels=self.in_features,
                        norm_layer=norm_layer,
                        activation=activation,
                        groups=refiner_groups[self.model_name],
                    ),
                )
            else:
                self.refiner = ConvUpsample(
                    in_channels=self.in_features,
                    out_channels=self.in_features,
                    norm_layer=norm_layer,
                    activation=activation,
                    groups=refiner_groups[self.model_name],
                )
        
        else:  # downsample
            assert block_size // patch_size == 2, f"Block size {block_size} should be 2 times the patch size {patch_size}."
            self.refiner = ConvDownsample(
                in_channels=self.in_features,
                out_channels=self.in_features,
                norm_layer=norm_layer,
                activation=activation,
                groups=refiner_groups[self.model_name],
            )
    
    def _adjust_pos_embed(self) -> Tensor:
        """
        Adjust the positional embedding to match the spatial resolution of the feature map.

        Args:
            orig_h, orig_w: The original spatial resolution of the image.
            new_h, new_w: The new spatial resolution of the image.
        """
        self.positional_embedding = nn.Parameter(self._interpolate_pos_embed(self.pretrain_size[0], self.pretrain_size[1], self.input_size[0], self.input_size[1]), requires_grad=False)

    def _interpolate_pos_embed(self, orig_h: int, orig_w: int, new_h: int, new_w: int) -> Tensor:
        """
        Interpolate the positional embedding to match the spatial resolution of the feature map.

        Args:
            orig_h, orig_w: The original spatial resolution of the image.
            new_h, new_w: The new spatial resolution of the image.
        """
        if (orig_h, orig_w) == (new_h, new_w):
            return self.positional_embedding
        
        orig_h_patches, orig_w_patches = orig_h // self.patch_size[0], orig_w // self.patch_size[1]
        new_h_patches, new_w_patches = new_h // self.patch_size[0], new_w // self.patch_size[1]
        class_pos_embed, patch_pos_embed = self.positional_embedding[:1, :], self.positional_embedding[1:, :]
        patch_pos_embed = rearrange(patch_pos_embed, "(h w) d -> d h w", h=orig_h_patches, w=orig_w_patches)
        patch_pos_embed = interpolate_pos_embed(patch_pos_embed, size=(new_h_patches, new_w_patches))
        patch_pos_embed = rearrange(patch_pos_embed, "d h w -> (h w) d")
        pos_embed = torch.cat((class_pos_embed, patch_pos_embed), dim=0)
        return pos_embed

    def train(self, mode: bool = True):
        if mode:
            # training:
            self.conv1.eval()
            self.ln_pre.eval()
            self.resblocks.eval()
            self.ln_post.eval()

            for idx in range(self.num_layers):
                getattr(self, f"vpt_drop_{idx}").train()

            self.refiner.train()

        else:
            # evaluation:
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
        x = self.conv1(x)

        # Step 2: Append the class token
        class_embedding = self.class_embedding.expand(batch_size, 1, -1)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = torch.cat([class_embedding, x], dim=1)

        # Step 3: Add positional embeddings
        pos_embed = self._interpolate_pos_embed(orig_h=self.input_size[0], orig_w=self.input_size[1], new_h=height, new_w=width).expand(batch_size, -1, -1)
        x = x + pos_embed
    
        x = self.ln_pre(x)
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
        x = self.resblocks[idx](x)

        # Disassemble
        x = torch.cat([
            x[:, :1, :],  # class token
            x[:, 1 + self.num_vpt:, :]  # patches
        ], dim=1)

        return x

    def _forward_adapter(self, x: Tensor, idx: int) -> Tensor:
        return getattr(self, f"adapter{idx}")(x)

    def forward_encoder(self, x: Tensor) -> Tensor:
        x = self._forward_patch_embed(x)
        for idx in range(self.num_layers):
            x = self._forward_adapter(x, idx) if self.adapter else self._forward_vpt(x, idx)
        x = self.ln_post(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        orig_h, orig_w = x.shape[-2:]
        num_patches_h, num_patches_w = orig_h // self.patch_size[0], orig_w // self.patch_size[1]
        x = self.forward_encoder(x)
        x = x[:, 1:, :]  # remove the class token
        x = rearrange(x, "b (h w) d -> b d h w", h=num_patches_h, w=num_patches_w)

        x = self.refiner(x)
        return x


def _vit(
    model_name: str,
    weight_name: str,
    block_size: int = 16,
    num_vpt: int = 32,
    vpt_drop: float = 0.1,
    adapter: bool = False,
    adapter_reduction: int = 4,
    lora: bool = False,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.1,
    input_size: Optional[Tuple[int, int]] = None,
    norm: str = "none",
    act: str = "none"
) -> ViT:
    assert not (lora and adapter), "LoRA and adapter cannot be used together."
    model = ViT(
        model_name=model_name,
        weight_name=weight_name,
        block_size=block_size,
        num_vpt=num_vpt,
        vpt_drop=vpt_drop,
        adapter=adapter,
        adapter_reduction=adapter_reduction,
        input_size=input_size,
        norm=norm,
        act=act
    )

    if lora:
        target_modules = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.MultiheadAttention)) and "refiner" not in name:
                target_modules.append(name)

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)

        # Unfreeze refiner
        for name, module in model.named_modules():
            if "refiner" in name:
                module.requires_grad_(True)

    return model
