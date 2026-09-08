import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict, Tuple
from copy import deepcopy

from .vit import vit_names_and_weights, _vit
from .convnext import convnext_names_and_weights, _convnext
from .resnet import resnet_names_and_weights, _resnet
from .mobileclip import mobileclip_names_and_weights, _mobileclip

from .utils import encode_text, optimize_text_prompts
from ..utils import conv1x1

supported_models_and_weights = deepcopy(vit_names_and_weights)
supported_models_and_weights.update(convnext_names_and_weights)
supported_models_and_weights.update(resnet_names_and_weights)
supported_models_and_weights.update(mobileclip_names_and_weights)


class CLIP_EBC(nn.Module):
    def __init__(
        self,
        model_name: str,
        weight_name: str,
        block_size: Optional[int] = None,
        bins: Optional[List[Tuple[float, float]]] = None,
        bin_centers: Optional[List[float]] = None,
        zero_inflated: Optional[bool] = True,
        num_vpt: Optional[int] = None,
        vpt_drop: Optional[float] = None,
        input_size: Optional[int] = None,
        adapter: Optional[bool] = False,
        adapter_reduction: Optional[int] = None,
        lora: Optional[bool] = False,
        lora_rank: Optional[int] = None,
        lora_alpha: Optional[float] = None,
        lora_dropout: Optional[float] = None,
        text_prompts: Optional[Dict[str, List[str]]] = None,
        norm: Optional[str] = "none",
        act: Optional[str] = "none",
    ) -> None:
        super().__init__()
        if "mobileclip" in model_name.lower() or "vit" in model_name.lower():
            model_name = model_name.replace("_", "-")
        assert model_name in supported_models_and_weights, f"Model name should be one of {list(supported_models_and_weights.keys())}, but got {model_name}."
        assert weight_name in supported_models_and_weights[model_name], f"Pretrained should be one of {supported_models_and_weights[model_name]}, but got {weight_name}."
        assert len(bins) == len(bin_centers), f"Expected bins and bin_centers to have the same length, got {len(bins)} and {len(bin_centers)}"
        assert len(bins) >= 2, f"Expected at least 2 bins, got {len(bins)}"
        assert all(len(b) == 2 for b in bins), f"Expected bins to be a list of tuples of length 2, got {bins}"
        bins = [(float(b[0]), float(b[1])) for b in bins]
        assert all(bin[0] <= p <= bin[1] for bin, p in zip(bins, bin_centers)), f"Expected bin_centers to be within the range of the corresponding bin, got {bins} and {bin_centers}"

        self.model_name = model_name
        self.weight_name = weight_name
        self.block_size = block_size
        self.bins = bins
        self.register_buffer("bin_centers", torch.tensor(bin_centers, dtype=torch.float32, requires_grad=False).view(1, -1, 1, 1))
        self.zero_inflated = zero_inflated
        self.text_prompts = text_prompts

        # Image encoder
        if model_name in vit_names_and_weights:
            assert num_vpt is not None and num_vpt >= 0, f"Number of VPT tokens should be greater than 0, but got {num_vpt}."
            vpt_drop = 0. if vpt_drop is None else vpt_drop
            self.backbone = _vit(
                model_name=model_name,
                weight_name=weight_name,
                num_vpt=num_vpt,
                vpt_drop=vpt_drop,
                block_size=block_size,
                adapter=adapter,
                adapter_reduction=adapter_reduction,
                lora=lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                input_size=(input_size, input_size),
                norm=norm,
                act=act
            )
        elif model_name in convnext_names_and_weights:
            self.backbone = _convnext(
                model_name=model_name,
                weight_name=weight_name,
                block_size=block_size,
                adapter=adapter,
                adapter_reduction=adapter_reduction,
                lora=lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                norm=norm,
                act=act
            )
        elif model_name in resnet_names_and_weights:
            self.backbone = _resnet(
                model_name=model_name,
                weight_name=weight_name,
                block_size=block_size,
                adapter=adapter,
                adapter_reduction=adapter_reduction,
                lora=lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                norm=norm,
                act=act
            )
        elif model_name in mobileclip_names_and_weights:
            self.backbone = _mobileclip(
                model_name=model_name,
                weight_name=weight_name,
                block_size=block_size,
                adapter=adapter,
                adapter_reduction=adapter_reduction,
                lora=lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                norm=norm,
                act=act
            )

        self._build_text_feats()
        self._build_head()

    def _build_text_feats(self) -> None:
        model_name, weight_name = self.model_name, self.weight_name
        text_prompts = self.text_prompts

        if text_prompts is None:
            bins = [b[0] if b[0] == b[1] else b for b in self.bins]  # if the bin is a single value (e.g., [0, 0]), use that value
            if self.zero_inflated:  # separate 0 from the rest
                assert bins[0] == 0, f"Expected the first bin to be 0, got {bins[0]}."
                bins_pi = [0, (1, float("inf"))]
                bins_lambda = bins[1:]
                pi_text_prompts = optimize_text_prompts(model_name, weight_name, bins_pi)
                lambda_text_prompts = optimize_text_prompts(model_name, weight_name, bins_lambda)
                self.text_prompts = {"pi": pi_text_prompts, "lambda": lambda_text_prompts}
                pi_text_feats = encode_text(model_name, weight_name, pi_text_prompts)
                lambda_text_feats = encode_text(model_name, weight_name, lambda_text_prompts)
                pi_text_feats.requires_grad = False
                lambda_text_feats.requires_grad = False
                self.register_buffer("pi_text_feats", pi_text_feats)
                self.register_buffer("lambda_text_feats", lambda_text_feats)

            else:
                text_prompts = optimize_text_prompts(model_name, weight_name, bins)
                self.text_prompts = text_prompts
                text_feats = encode_text(model_name, weight_name, text_prompts)
                text_feats.requires_grad = False
                self.register_buffer("text_feats", text_feats)
        
        else:
            if self.zero_inflated:
                assert "pi" in text_prompts and "lambda" in text_prompts, f"Expected text_prompts to have keys 'pi' and 'lambda', got {text_prompts.keys()}."
                pi_text_prompts = text_prompts["pi"]
                lambda_text_prompts = text_prompts["lambda"]
                pi_text_feats = encode_text(model_name, weight_name, pi_text_prompts)
                lambda_text_feats = encode_text(model_name, weight_name, lambda_text_prompts)
                pi_text_feats.requires_grad = False
                lambda_text_feats.requires_grad = False
                self.register_buffer("pi_text_feats", pi_text_feats)
                self.register_buffer("lambda_text_feats", lambda_text_feats)

            else:
                text_feats = encode_text(model_name, weight_name, text_prompts)
                text_feats.requires_grad = False
                self.register_buffer("text_feats", text_feats)

    def _build_head(self) -> None:
        in_channels = self.backbone.in_features
        out_channels = self.backbone.out_features
        if self.zero_inflated:
            self.pi_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)
            self.lambda_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)

            self.pi_head = conv1x1(in_channels, out_channels, bias=False)
            self.lambda_head = conv1x1(in_channels, out_channels, bias=False)

        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)
            self.head = conv1x1(in_channels, out_channels, bias=False)

    def forward(self, image: Tensor):
        image_feats = self.backbone(image)
        # image_feats = F.normalize(image_feats.permute(0, 2, 3, 1), p=2, dim=-1)  # shape (B, H, W, C)
        
        if self.zero_inflated:
            pi_image_feats, lambda_image_feats = self.pi_head(image_feats), self.lambda_head(image_feats)
            pi_image_feats = F.normalize(pi_image_feats.permute(0, 2, 3, 1), p=2, dim=-1)  # shape (B, H, W, C)
            lambda_image_feats = F.normalize(lambda_image_feats.permute(0, 2, 3, 1), p=2, dim=-1)  # shape (B, H, W, C)

            pi_text_feats, lambda_text_feats = self.pi_text_feats, self.lambda_text_feats
            pi_logit_scale, lambda_logit_scale = self.pi_logit_scale.exp(), self.lambda_logit_scale.exp()

            pi_logit_map = pi_logit_scale * pi_image_feats @ pi_text_feats.t()  # (B, H, W, 2), logits per image
            lambda_logit_map = lambda_logit_scale * lambda_image_feats @ lambda_text_feats.t()  # (B, H, W, N - 1), logits per image

            pi_logit_map =  pi_logit_map.permute(0, 3, 1, 2)  # (B, 2, H, W)
            lambda_logit_map = lambda_logit_map.permute(0, 3, 1, 2)  # (B, N - 1, H, W)

            lambda_map = (lambda_logit_map.softmax(dim=1) * self.bin_centers[:, 1:]).sum(dim=1, keepdim=True)  # (B, 1, H, W)
            
            # pi_logit_map.softmax(dim=1)[:, 0] is the probability of zeros
            den_map = pi_logit_map.softmax(dim=1)[:, 1:] * lambda_map # (B, 1, H, W)
            
            if self.training:
                return pi_logit_map, lambda_logit_map, lambda_map, den_map
            else:
                return den_map
        
        else:
            image_feats = self.head(image_feats)
            image_feats = F.normalize(image_feats.permute(0, 2, 3, 1), p=2, dim=-1)

            text_feats = self.text_feats
            logit_scale = self.logit_scale.exp()

            logit_map = logit_scale * image_feats @ text_feats.t()  # (B, H, W, N), logits per image
            logit_map = logit_map.permute(0, 3, 1, 2)  # (B, N, H, W)

            den_map = (logit_map.softmax(dim=1) * self.bin_centers).sum(dim=1, keepdim=True)  # (B, 1, H, W)

            if self.training:
                return logit_map, den_map
            else:
                return den_map


def _clip_ebc(
    model_name: str,
    weight_name: str,
    block_size: Optional[int] = None,
    bins: Optional[List[Tuple[float, float]]] = None,
    bin_centers: Optional[List[float]] = None,
    zero_inflated: Optional[bool] = True,
    num_vpt: Optional[int] = None,
    vpt_drop: Optional[float] = None,
    input_size: Optional[int] = None,
    adapter: Optional[bool] = False,
    adapter_reduction: Optional[int] = None,
    lora: Optional[bool] = False,
    lora_rank: Optional[int] = None,
    lora_alpha: Optional[float] = None,
    lora_dropout: Optional[float] = None,
    text_prompts: Optional[List[str]] = None,
    norm: Optional[str] = "none",
    act: Optional[str] = "none",
) -> CLIP_EBC:
    return CLIP_EBC(
        model_name=model_name,
        weight_name=weight_name,
        block_size=block_size,
        bins=bins,
        bin_centers=bin_centers,
        zero_inflated=zero_inflated,
        num_vpt=num_vpt,
        vpt_drop=vpt_drop,
        input_size=input_size,
        adapter=adapter,
        adapter_reduction=adapter_reduction,
        lora=lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        text_prompts=text_prompts,
        norm=norm,
        act=act,
    )