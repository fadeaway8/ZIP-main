import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from typing import Callable, Optional, Sequence, Tuple, Union, List, List
import warnings

from .utils import _init_weights, _make_ntuple, _log_api_usage_once


def conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    bias: bool = True,
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )
    conv.apply(_init_weights)
    return conv


def conv1x1(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    bias: bool = True,
) -> nn.Conv2d:
    """1x1 convolution"""
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)
    conv.apply(_init_weights)
    return conv


class DepthSeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        # Depthwise convolution: one filter per input channel.
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode
        )
        # Pointwise convolution: combine the features across channels.
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=bias,
            padding_mode=padding_mode
        )
        self.apply(_init_weights)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        self.apply(_init_weights)

    def forward(self, x: Tensor) -> Tensor:
        B, C, _, _ = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Union[nn.BatchNorm2d, nn.GroupNorm, None] = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU(inplace=True),
        groups: int = 1,
    ) -> None:
        super().__init__()
        assert isinstance(groups, int) and groups > 0, f"Expected groups to be a positive integer, but got {groups}"
        assert in_channels % groups == 0, f"Expected in_channels to be divisible by groups, but got {in_channels} % {groups}"
        assert out_channels % groups == 0, f"Expected out_channels to be divisible by groups, but got {out_channels} % {groups}"
        self.grouped_conv = groups > 1
        self.conv1 = conv3x3(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=1,
            bias=not norm_layer,
            groups=groups,
        )
        if self.grouped_conv:
            self.conv1_1x1 = conv1x1(out_channels, out_channels, stride=1, bias=not norm_layer)
        
        self.norm1 = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.act1 = activation

        self.conv2 = conv3x3(
            in_channels=out_channels,
            out_channels=out_channels,
            stride=1,
            bias=not norm_layer,
            groups=groups,
        )
        if self.grouped_conv:
            self.conv2_1x1 = conv1x1(out_channels, out_channels, stride=1, bias=not norm_layer)
        
        self.norm2 = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.act2 = activation

        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride=1, bias=not norm_layer),
                norm_layer(out_channels) if norm_layer else nn.Identity(),
            )
        else:
            self.downsample = nn.Identity()
        
        self.apply(_init_weights)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv1_1x1(out) if self.grouped_conv else out
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.conv2_1x1(out) if self.grouped_conv else out
        out = self.norm2(out)

        out += self.downsample(identity)
        out = self.act2(out)

        return out


class LightBasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Union[nn.BatchNorm2d, nn.GroupNorm, None] = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU(inplace=True),
    ) -> None:
        super().__init__()
        self.conv1 = DepthSeparableConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not norm_layer,
        )
        self.norm1 = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.act1 = activation

        self.conv2 = DepthSeparableConv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not norm_layer,
        )
        self.norm2 = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.act2 = activation

        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride=1, bias=not norm_layer),
                norm_layer(out_channels) if norm_layer else nn.Identity(),
            )
        else:
            self.downsample = nn.Identity()
        
        self.apply(_init_weights)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += self.downsample(identity)
        out = self.act2(out)

        return out


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Union[nn.BatchNorm2d, nn.GroupNorm, None] = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU(inplace=True),
        groups: int = 1,
        base_width: int = 64,
        expansion: float = 2.0,
    ) -> None:
        super().__init__()
        assert isinstance(groups, int) and groups > 0, f"Expected groups to be a positive integer, but got {groups}"
        assert expansion > 0, f"Expected expansion to be greater than 0, but got {expansion}"
        assert base_width > 0, f"Expected base_width to be greater than 0, but got {base_width}"
        bottleneck_channels = int(in_channels * (base_width / 64.0) * expansion)
        assert bottleneck_channels % groups == 0, f"Expected bottleneck_channels to be divisible by groups, but got {bottleneck_channels} % {groups}"
        self.grouped_conv = groups > 1
        self.expansion, self.base_width = expansion, base_width

        self.conv_in = conv1x1(in_channels, bottleneck_channels, stride=1, bias=not norm_layer)
        self.norm_in = norm_layer(bottleneck_channels)
        self.act_in = activation

        self.se_in = SEBlock(bottleneck_channels) if bottleneck_channels > in_channels else nn.Identity()

        self.conv_block_1 = nn.Sequential(
            conv3x3(
                in_channels=bottleneck_channels,
                out_channels=bottleneck_channels,
                stride=1,
                groups=groups,
                bias=not norm_layer
            ),
            conv1x1(bottleneck_channels, bottleneck_channels, stride=1, bias=not norm_layer) if groups > 1 else nn.Identity(),
            norm_layer(bottleneck_channels) if norm_layer else nn.Identity(),
            activation,
        )

        self.conv_block_2 = nn.Sequential(
            conv3x3(
                in_channels=bottleneck_channels,
                out_channels=bottleneck_channels,
                stride=1,
                groups=groups,
                bias=not norm_layer
            ),
            conv1x1(bottleneck_channels, bottleneck_channels, stride=1, bias=not norm_layer) if groups > 1 else nn.Identity(),
            norm_layer(bottleneck_channels) if norm_layer else nn.Identity(),
            activation,
        )

        self.conv_out = conv1x1(bottleneck_channels, out_channels, stride=1, bias=not norm_layer)
        self.norm_out = norm_layer(out_channels)
        self.act_out = activation
        self.se_out = SEBlock(out_channels) if out_channels > bottleneck_channels else nn.Identity()

        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride=1, bias=not norm_layer),
                norm_layer(out_channels) if norm_layer else nn.Identity(),
            )
        else:
            self.downsample = nn.Identity()

        self.apply(_init_weights)
    
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # expand
        out = self.conv_in(x)
        out = self.norm_in(out)
        out = self.act_in(out)
        out = self.se_in(out)

        # conv
        out = self.conv_block_1(out)
        out = self.conv_block_2(out)

        # reduce
        out = self.conv_out(out)
        out = self.norm_out(out)
        out = self.se_out(out)

        out += self.downsample(identity)
        out = self.act_out(out)
        return out
    

class ConvASPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilations: List[int] = [1, 2, 4],
        norm_layer: Union[nn.BatchNorm2d, nn.GroupNorm, None] = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU(inplace=True),
        groups: int = 1,
        base_width: int = 64,
        expansion: float = 2.0,
    ) -> None:
        super().__init__()
        assert isinstance(groups, int) and groups > 0, f"Expected groups to be a positive integer, but got {groups}"
        assert expansion > 0, f"Expected expansion to be greater than 0, but got {expansion}"
        assert base_width > 0, f"Expected base_width to be greater than 0, but got {base_width}"
        bottleneck_channels = int(in_channels * (base_width / 64.0) * expansion)
        assert bottleneck_channels % groups == 0, f"Expected bottleneck_channels to be divisible by groups, but got {bottleneck_channels} % {groups}"
        self.expansion, self.base_width = expansion, base_width

        self.conv_in = conv1x1(in_channels, bottleneck_channels, stride=1, bias=not norm_layer)
        self.norm_in = norm_layer(bottleneck_channels)
        self.act_in = activation

        conv_blocks = [nn.Sequential(
            conv1x1(bottleneck_channels, bottleneck_channels, stride=1, bias=not norm_layer),
            norm_layer(bottleneck_channels),
            activation
        )]
    
        for dilation in dilations:
            conv_blocks.append(nn.Sequential(
                conv3x3(
                in_channels=bottleneck_channels,
                out_channels=bottleneck_channels,
                stride=1,
                groups=groups,
                dilation=dilation,
                bias=not norm_layer
            ),
            conv1x1(bottleneck_channels, bottleneck_channels, stride=1, bias=not norm_layer) if groups > 1 else nn.Identity(),
                norm_layer(bottleneck_channels) if norm_layer else nn.Identity(),
                activation
            ))
        
        self.convs = nn.ModuleList(conv_blocks)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_avg = conv1x1(bottleneck_channels, bottleneck_channels, stride=1, bias=not norm_layer)
        self.norm_avg = norm_layer(bottleneck_channels)
        self.act_avg = activation

        self.se = SEBlock(bottleneck_channels * (len(dilations) + 2))

        self.conv_out = conv1x1(bottleneck_channels * (len(dilations) + 2), out_channels, stride=1, bias=not norm_layer)
        self.norm_out = norm_layer(out_channels)
        self.act_out = activation

        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride=1, bias=not norm_layer),
                norm_layer(out_channels) if norm_layer else nn.Identity(),
            )
        else:
            self.downsample = nn.Identity()

        self.apply(_init_weights)

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        identity = x

        # expand
        out = self.conv_in(x)
        out = self.norm_in(out)
        out = self.act_in(out)

        outs = []
        for conv in self.convs:
            outs.append(conv(out))
        
        avg = self.avgpool(out)
        avg = self.conv_avg(avg)
        avg = self.norm_avg(avg)
        avg = self.act_avg(avg)  # (B, C, 1, 1)
        avg = avg.repeat(1, 1, height, width)

        outs = torch.cat([*outs, avg], dim=1)  # (B, C * (len(dilations) + 2), H, W)
        outs = self.se(outs)

        # reduce
        outs = self.conv_out(outs)
        outs = self.norm_out(outs)

        outs += self.downsample(identity)
        outs = self.act_out(outs)
        return outs


class ViTBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, f"Embedding dimension {embed_dim} should be divisible by number of heads {num_heads}"
        self.embed_dim, self.num_heads  = embed_dim, num_heads
        self.dropout, self.mlp_ratio = dropout, mlp_ratio

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )
        self.apply(_init_weights)
    
    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 3, f"Expected input to have shape (B, N, C), but got {x.shape}"
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Conv2dLayerNorm(nn.Sequential):
    """
    Layer normalization applied in a convolutional fashion.
    """
    def __init__(self, dim: int) -> None:
        super().__init__(
            Rearrange("B C H W -> B H W C"),
            nn.LayerNorm(dim),
            Rearrange("B H W C -> B C H W")
        )
        self.apply(_init_weights)


class CvTAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        q_stride: int = 1,  # controls downsampling rate
        kv_stride: int = 1,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, f"Embedding dimension {embed_dim} should be divisible by number of heads {num_heads}"
        self.embed_dim, self.num_heads, self.dim_head = embed_dim, num_heads, embed_dim // num_heads
        self.scale = self.dim_head ** -0.5
        self.q_stride, self.kv_stride = q_stride, kv_stride
        
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = DepthSeparableConv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=3,
            stride=q_stride,
            padding=1,
            bias=False
        )
        self.to_k = DepthSeparableConv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=3,
            stride=kv_stride,
            padding=1,
            bias=False
        )
        self.to_v = DepthSeparableConv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=3,
            stride=kv_stride,
            padding=1,
            bias=False
        )

        self.to_out = nn.Sequential(
            conv1x1(embed_dim, embed_dim, stride=1),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

        self.apply(_init_weights)
    
    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 4, f"Expected input to have shape (B, C, H, W), but got {x.shape}"
        assert x.shape[1] == self.embed_dim, f"Expected input to have embedding dimension {self.embed_dim}, but got {x.shape[1]}"
        
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        B, _, H, W = q.shape
        q, k, v = map(lambda t: rearrange(t, "B (num_heads head_dim) H W -> (B num_heads) (H W) head_dim", num_heads=self.num_heads), (q, k, v))
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attend(attn)
        attn = self.dropout(attn)

        out = attn @ v
        out = rearrange(out, "(B num_heads) (H W) head_dim -> B (num_heads head_dim) H W", B=B, H=H, W=W, num_heads=self.num_heads)
        out = self.to_out(out)

        return out


class CvTBlock(nn.Module):
    """
    Implement convolutional vision transformer block.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        q_stride: int = 1,
        kv_stride: int = 1,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, f"Embedding dimension {embed_dim} should be divisible by number of heads {num_heads}."
        self.embed_dim, self.num_heads = embed_dim, num_heads

        self.norm1 = Conv2dLayerNorm(embed_dim)
        self.attn = CvTAttention(embed_dim, num_heads, dropout, q_stride, kv_stride)

        self.pool = nn.AvgPool2d(kernel_size=q_stride, stride=q_stride) if q_stride > 1 else nn.Identity()
        
        self.norm2 = Conv2dLayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(embed_dim, int(embed_dim * mlp_ratio), kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(int(embed_dim * mlp_ratio), embed_dim, kernel_size=1),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(x) + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    

class ConvAdapter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int = 16,
    ) -> None:
        super().__init__()
        assert in_channels > 0, f"Expected input_channels to be greater than 0, but got {in_channels}"
        assert bottleneck_channels > 0, f"Expected bottleneck_channels to be greater than 0, but got {bottleneck_channels}"
        
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(bottleneck_channels, in_channels, kernel_size=1),
        )
        nn.init.zeros_(self.adapter[2].weight)
        nn.init.zeros_(self.adapter[2].bias)
    
    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 4, f"Expected input to have shape (B, C, H, W), but got {x.shape}"
        return x + self.adapter(x)


class ViTAdapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.GELU(), # ViT中常用GELU作为激活函数
            nn.Linear(bottleneck_dim, input_dim)
        )
        nn.init.zeros_(self.adapter[2].weight)
        nn.init.zeros_(self.adapter[2].bias)
    
    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 3, f"Expected input to have shape (B, N, C), but got {x.shape}"
        return x + self.adapter(x)
        