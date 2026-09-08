import torch
from torch import nn, Tensor
from einops import rearrange
from typing import List, Tuple
from .utils import _reshape_density, _bin_count

EPS = 1e-8


class ZIPoissonNLL(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        assert reduction in ["none", "mean", "sum"], f"Expected reduction to be one of ['none', 'mean', 'sum'], got {reduction}."
        self.reduction = reduction

    def forward(
        self,
        logit_pi_maps: Tensor,
        lambda_maps: Tensor,
        gt_den_maps: Tensor,
    ) -> Tensor:
        assert len(logit_pi_maps.shape) == len(lambda_maps.shape) == len(gt_den_maps.shape) == 4, f"Expected 4D (B, C, H, W) tensor, got {logit_pi_maps.shape}, {lambda_maps.shape}, and {gt_den_maps.shape}"
        B, _, H, W = lambda_maps.shape
        assert logit_pi_maps.shape == (B, 2, H, W), f"Expected logit_pi_maps to have shape (B, 2, H, W), got {logit_pi_maps.shape}"
        assert lambda_maps.shape == (B, 1, H, W), f"Expected lambda_maps to have shape (B, 1, H, W), got {lambda_maps.shape}"
        if gt_den_maps.shape[2:] != (H, W):
            gt_h, gt_w = gt_den_maps.shape[-2], gt_den_maps.shape[-1]
            assert gt_h % H == 0 and gt_w % W == 0 and gt_h // H == gt_w // W, f"Expected the spatial dimension of gt_den_maps to be a multiple of that of lambda_maps, got {gt_den_maps.shape} and {lambda_maps.shape}"
            gt_den_maps = _reshape_density(gt_den_maps, block_size=gt_h // H)
        assert gt_den_maps.shape == (B, 1, H, W), f"Expected gt_den_maps to have shape (B, 1, H, W), got {gt_den_maps.shape}"

        pi_maps = logit_pi_maps.softmax(dim=1)
        zero_indices = (gt_den_maps == 0).float()
        zero_loss = -torch.log(pi_maps[:, 0:1] + pi_maps[:, 1:] * torch.exp(-lambda_maps) + EPS) * zero_indices

        poisson_log_p = gt_den_maps * torch.log(lambda_maps + EPS) - lambda_maps
        nonzero_loss = (-torch.log(pi_maps[:, 1:] + EPS) - poisson_log_p) * (1.0 - zero_indices)

        loss = (zero_loss + nonzero_loss).sum(dim=(-1, -2))  
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss, {"zipnll": loss.detach()}


class ZICrossEntropy(nn.Module):
    def __init__(
        self,
        bins: List[Tuple[int, int]],
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        assert all([low <= high for low, high in bins]), f"Expected bins to be a list of tuples (low, high) where low <= high, got {bins}"
        assert reduction in ["mean", "sum"], f"Expected reduction to be one of ['none', 'mean', 'sum'], got {reduction}."

        self.bins = bins
        self.reduction = reduction
        self.ce_loss_fn = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        logit_maps: Tensor,
        gt_den_maps: Tensor,
    ) -> Tensor:
        assert len(logit_maps.shape) == len(gt_den_maps.shape) == 4, f"Expected 4D (B, C, H, W) tensor, got {logit_maps.shape} and {gt_den_maps.shape}"
        B, _, H, W = logit_maps.shape
        assert logit_maps.shape[0] == B and logit_maps.shape[2:] == (H, W), f"Expected logit_maps to have shape (B, C, H, W), got {logit_maps.shape}"
        if gt_den_maps.shape[2:] != (H, W):
            gt_h, gt_w = gt_den_maps.shape[-2], gt_den_maps.shape[-1]
            assert gt_h % H == 0 and gt_w % W == 0 and gt_h // H == gt_w // W, f"Expected the spatial dimension of gt_den_maps to be a multiple of that of logit_maps, got {gt_den_maps.shape} and {logit_maps.shape}"
            gt_den_maps = _reshape_density(gt_den_maps, block_size=gt_h // H)
        assert gt_den_maps.shape == (B, 1, H, W), f"Expected gt_den_maps to have shape (B, 1, H, W), got {gt_den_maps.shape}"

        gt_class_maps = _bin_count(gt_den_maps, bins=self.bins)
        gt_class_maps = rearrange(gt_class_maps, "B H W -> B (H W)")  # flatten spatial dimensions
        logit_maps = rearrange(logit_maps, "B C H W -> B (H W) C")  # flatten spatial dimensions

        loss = 0.0
        for idx in range(gt_class_maps.shape[0]):
            gt_class_map, logit_map = gt_class_maps[idx], logit_maps[idx]
            mask = gt_class_map > 0
            # Find gt_class_map values and logit_maps values where gt_class_map > 0
            gt_class_map = gt_class_map[mask] - 1
            logit_map = logit_map[mask]
            loss += self.ce_loss_fn(logit_map, gt_class_map).sum()

        if self.reduction == "mean":
            loss /= gt_class_maps.shape[0]
        
        return loss, {"cls_zice": loss.detach()}
        