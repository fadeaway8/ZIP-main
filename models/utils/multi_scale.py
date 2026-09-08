import torch
from torch import nn, Tensor
from typing import List
from einops import rearrange

from .blocks import conv3x3, conv1x1, Conv2dLayerNorm, _init_weights


class MultiScale(nn.Module):
    def __init__(
        self,
        channels: int,
        scales: List[int],
        heads: int = 8,
        groups: int = 1,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        assert channels > 0, "channels should be a positive integer"
        assert isinstance(scales, (list, tuple)) and len(scales) > 0 and all([scale > 0 for scale in scales]), "scales should be a list or tuple of positive integers"
        assert heads > 0 and channels % heads == 0, "heads should be a positive integer and channels should be divisible by heads"
        assert groups > 0 and channels % groups == 0, "groups should be a positive integer and channels should be divisible by groups"
        scales = sorted(scales)
        self.scales = scales
        self.num_scales = len(scales) + 1  # +1 for the original feature map
        self.heads = heads
        self.groups = groups

        # modules that generate multi-scale feature maps
        self.scale_0 = nn.Sequential(
            conv1x1(channels, channels, stride=1, bias=False),
            Conv2dLayerNorm(channels),
            nn.GELU(),
        )
        for scale in scales:
            setattr(self, f"conv_{scale}", nn.Sequential(
                conv3x3(
                    in_channels=channels,
                    out_channels=channels,
                    stride=1,
                    groups=groups,
                    dilation=scale,
                    bias=False,
                ),
                conv1x1(channels, channels, stride=1, bias=False) if groups > 1 else nn.Identity(),
                Conv2dLayerNorm(channels),
                nn.GELU(),
            ))
        
        # modules that fuse multi-scale feature maps
        self.norm_attn = Conv2dLayerNorm(channels)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_scales + 1, channels, 1, 1) / channels ** 0.5)
        self.to_q = conv1x1(channels, channels, stride=1, bias=False)
        self.to_k = conv1x1(channels, channels, stride=1, bias=False)
        self.to_v = conv1x1(channels, channels, stride=1, bias=False)

        self.scale = (channels // heads) ** -0.5

        self.attend = nn.Softmax(dim=-1)
        
        self.to_out = conv1x1(channels, channels, stride=1)

        # modules that refine multi-scale feature maps
        self.norm_mlp = Conv2dLayerNorm(channels)
        self.mlp = nn.Sequential(
            conv1x1(channels, channels * mlp_ratio, stride=1),
            nn.GELU(),
            conv1x1(channels * mlp_ratio, channels, stride=1),
        )
        
        self.apply(_init_weights)
    
    def _forward_attn(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 4, f"Expected input to have shape (B, C, H, W), but got {x.shape}"
        x = [self.scale_0(x)] + [getattr(self, f"conv_{scale}")(x) for scale in self.scales]

        x = torch.stack(x, dim=1)  # (B, S, C, H, W)
        x = torch.cat([x.mean(dim=1, keepdim=True), x], dim=1)  # (B, S+1, C, H, W)
        x = x + self.pos_embed  # (B, S+1, C, H, W)

        x = rearrange(x, "B S C H W -> (B S) C H W")  # (B*(S+1), C, H, W)
        x = self.norm_attn(x)  # (B*(S+1), C, H, W)
        x = rearrange(x, "(B S) C H W -> B S C H W", S=self.num_scales + 1)  # (B, S+1, C, H, W)
        
        q = self.to_q(x[:, 0])  # (B, C, H, W)
        k = self.to_k(rearrange(x, "B S C H W -> (B S) C H W"))
        v = self.to_v(rearrange(x, "B S C H W -> (B S) C H W"))

        q = rearrange(q, "B (h d) H W -> B h H W 1 d", h=self.heads)
        k = rearrange(k, "(B S) (h d) H W -> B h H W S d", S=self.num_scales + 1, h=self.heads)
        v = rearrange(v, "(B S) (h d) H W -> B h H W S d", S=self.num_scales + 1, h=self.heads)

        attn = q @ k.transpose(-2, -1) * self.scale  # (B, h, H, W, 1, S+1)
        attn = self.attend(attn)  # (B, h, H, W, 1, S+1)
        out = attn @ v  # (B, h, H, W, 1, d)

        out = rearrange(out, "B h H W 1 d -> B (h d) H W")  # (B, C, H, W)

        out = self.to_out(out)  # (B, C, H, W)
        return out
    
    def _forward_mlp(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 4, f"Expected input to have shape (B, C, H, W), but got {x.shape}"
        x = self.norm_mlp(x)
        x = self.mlp(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = x + self._forward_attn(x)
        x = x + self._forward_mlp(x)
        return x
    