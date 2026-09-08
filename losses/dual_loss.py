import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List, Tuple, Dict

from .dm_loss import DMLoss
from .multiscale_mae import MultiscaleMAE
from .utils import _reshape_density



class DualLoss(nn.Module):
    def __init__(
        self,
        input_size: int,
        block_size: int,
        bins: List[Tuple[float, float]],
        bin_centers: List[float],
        cls_loss: str = "ce",
        reg_loss: str = "dm",
        weight_tv: float = 0.01,
        weight_cls: float = 0.1,
        weight_reg: float = 0.1,
        numItermax: int = 100,
        regularization: float = 10.0,
        scales: List[int] = [1, 2, 4],
        min_scale_weight: float = 0.25,
        max_scale_weight: float = 0.75,
        alpha: float = 0.5,
    ) -> None:
        super().__init__()
        assert len(bins) == len(bin_centers) >= 2, f"Expected bins and bin_centers to have at least 2 elements, got {len(bins)} and {len(bin_centers)}"
        assert all([len(b) == 2 for b in bins]), f"Expected all bins to be of length 2, got {bins}"
        assert all(b[0] <= p <= b[1] for b, p in zip(bins, bin_centers)), f"Expected bin_centers to be within the range of the corresponding bin, got {bins} and {bin_centers}"
        assert cls_loss in ["ce", "mae", "mse", "none"], f"Expected cls_loss to be one of ['ce', 'mae', 'mse', 'none'], got {cls_loss}"
        assert reg_loss in ["dm", "msmae", "mae", "mse", "none"], f"Expected reg_loss to be one of ['dm', 'msmae', 'mae', 'mse', 'none'], got {reg_loss}"
        assert not (cls_loss == "none" and reg_loss == "none"), "Expected at least one of cls_loss and reg_loss to be provided"
        assert weight_cls is None or weight_cls >= 0, f"Expected weight_cls to be non-negative, got {weight_cls}"
        assert weight_reg is None or weight_reg >= 0, f"Expected weight_reg to be non-negative, got {weight_reg}"
        assert weight_tv is None or weight_tv >= 0, f"Expected weight_tv to be non-negative, got {weight_tv}"
        assert min_scale_weight is None or max_scale_weight is None or max_scale_weight >= min_scale_weight > 0, f"Expected max_scale_weight to be greater than or equal to min_scale_weight, got {min_scale_weight} and {max_scale_weight}"
        assert alpha is None or 1 > alpha > 0, f"Expected alpha to be between 0 and 1, got {alpha}"

        if reg_loss == "dm":
            assert numItermax is not None and numItermax > 0, f"Expected numItermax to be a positive integer, got {numItermax}"
            assert regularization is not None and regularization > 0, f"Expected regularization to be a positive float, got {regularization}"
            assert weight_tv is not None and weight_tv >= 0, f"Expected weight_tv to be non-negative, got {weight_tv}"
        else:
            weight_tv, numItermax, regularization = None, None, None
        
        if reg_loss == "msmae":
            assert isinstance(scales, (list, tuple)) and len(scales) > 0 and all(isinstance(s, int) and s > 0 for s in scales), f"Expected scales to be a list of positive integers, got {scales}"
            assert max_scale_weight >= min_scale_weight > 0, f"Expected max_scale_weight to be greater than or equal to min_scale_weight, got {min_scale_weight} and {max_scale_weight}"
            assert 1 > alpha > 0, f"Expected alpha to be between 0 and 1, got {alpha}"
        else:
            scales = None
            min_scale_weight, max_scale_weight = None, None
            alpha = None

        weight_cls = weight_cls if weight_cls is not None else 0
        weight_reg = weight_reg if weight_reg is not None else 0

        self.input_size, self.block_size = input_size, block_size
        self.num_blocks_h, self.num_blocks_w = input_size // block_size, input_size // block_size
        self.bins, self.bin_centers, self.num_bins = bins, bin_centers, len(bins)
        self.cls_loss, self.reg_loss = cls_loss, reg_loss
        self.weight_cls, self.weight_reg = weight_cls, weight_reg
        self.numItermax, self.regularization = numItermax, regularization
        self.weight_tv = weight_tv
        self.scales = scales
        self.min_scale_weight, self.max_scale_weight = min_scale_weight, max_scale_weight

        if cls_loss == "ce":
            self.cls_loss_fn = nn.CrossEntropyLoss(reduction="none")
            self.weight_cls = 1.0
        elif cls_loss == "mae":
            self.cls_loss_fn = nn.L1Loss(reduction="none")
            self.weight_cls = weight_cls
        elif cls_loss == "mse":
            self.cls_loss_fn = nn.MSELoss(reduction="none")
            self.weight_cls = weight_cls
        else:  # cls_loss == "none"
            self.cls_loss_fn = None
            self.weight_cls = 0

        if reg_loss == "dm":
            self.reg_loss_fn = DMLoss(
                input_size=input_size,
                block_size=block_size,
                numItermax=numItermax,
                regularization=regularization,
                weight_ot=weight_reg,
                weight_tv=weight_tv,
                weight_cnt=0,  # Calculate the count loss separately
            )
            self.weight_reg = 1.0
        elif reg_loss == "msmae":
            self.reg_loss_fn = MultiscaleMAE(scales=scales, weights=None, min_scale_weight=min_scale_weight, max_scale_weight=max_scale_weight, alpha=alpha)
            self.weight_reg = 1.0
        elif reg_loss == "mae":
            self.reg_loss_fn = nn.L1Loss(reduction="none")
            self.weight_reg = weight_reg
        elif reg_loss == "mse":
            self.reg_loss_fn = nn.MSELoss(reduction="none")
            self.weight_reg = weight_reg
        else:
            self.reg_loss_fn = None
            self.weight_reg = 0
        
        self.cnt_loss_fn = nn.L1Loss(reduction="none")
        
    def _bin_count(self, density_map: Tensor) -> Tensor:
        class_map = torch.zeros_like(density_map, dtype=torch.long)
        for idx, (low, high) in enumerate(self.bins):
            mask = (density_map >= low) & (density_map <= high)
            class_map[mask] = idx
        return class_map.squeeze(1)  # remove channel dimension

    def forward(
        self,
        pred_logit_map: Tensor,
        pred_den_map: Tensor,
        gt_den_map: Tensor,
        gt_points: List[Tensor]
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        B = pred_logit_map.shape[0]
        assert pred_logit_map.shape == (B, self.num_bins, self.num_blocks_h, self.num_blocks_w), f"Expected pred_logit_map to have shape {B, self.num_bins, self.num_blocks_h, self.num_blocks_w}, got {pred_logit_map.shape}"
        if gt_den_map.shape[-2:] != (self.num_blocks_h, self.num_blocks_w):
            assert gt_den_map.shape[-2:] == (self.input_size, self.input_size), f"Expected gt_den_map to have shape {B, 1, self.input_size, self.input_size}, got {gt_den_map.shape}"
            gt_den_map = _reshape_density(gt_den_map, block_size=self.block_size)
        assert pred_den_map.shape == gt_den_map.shape == (B, 1, self.num_blocks_h, self.num_blocks_w), f"Expected pred_den_map and gt_den_map to have shape (B, 1, H, W), got {pred_den_map.shape} and {gt_den_map.shape}"
        assert len(gt_points) == B, f"Expected gt_points to have length B, got {len(gt_points)}"
        
        loss_info = {}

        if self.weight_cls > 0:
            gt_class_map = self._bin_count(gt_den_map)
            if self.cls_loss == "ce":
                cls_loss = self.cls_loss_fn(pred_logit_map, gt_class_map).sum(dim=(-1, -2)).mean()
                loss_info["cls_ce_loss"] = cls_loss.detach()
            else:  # self.cls_loss in ["mae", "mse"]
                gt_prob_map = F.one_hot(gt_class_map, num_classes=self.num_bins).float()  # B, H, W -> B, H, W, N
                gt_prob_map = gt_prob_map.permute(0, 3, 1, 2)  # B, H, W, N -> B, N, H, W
                pred_prob_map = pred_logit_map.softmax(dim=1)
                cls_loss = self.cls_loss_fn(pred_prob_map, gt_prob_map).sum(dim=(-1, -2)).mean()
                loss_info[f"cls_{self.cls_loss}_loss"] = cls_loss.detach()
        else:
            cls_loss = 0

        if self.weight_reg > 0:
            if self.reg_loss == "dm":
                reg_loss, reg_loss_info = self.reg_loss_fn(
                    pred_den_map=pred_den_map,
                    gt_den_map=gt_den_map,
                    gt_points=gt_points,
                )
                loss_info.update({f"reg_{k}": v for k, v in reg_loss_info.items()})
            elif self.reg_loss == "msmae":
                reg_loss, reg_loss_info = self.reg_loss_fn(pred_den_map, gt_den_map)
                loss_info.update({f"reg_{k}": v for k, v in reg_loss_info.items()})
            else:  # self.reg_loss in ["mae", "mse"]
                reg_loss = self.reg_loss_fn(pred_den_map, gt_den_map).sum(dim=(-1, -2)).mean()
                loss_info[f"reg_{self.reg_loss}_loss"] = reg_loss.detach()
        else:
            reg_loss = 0

        gt_cnt = torch.tensor([len(p) for p in gt_points], dtype=torch.float32, device=pred_den_map.device)
        cnt_loss = self.cnt_loss_fn(pred_den_map.sum(dim=(1, 2, 3)), gt_cnt).mean()
        loss_info["cnt_loss"] = cnt_loss.detach()

        total_loss = self.weight_cls * cls_loss + self.weight_reg * reg_loss + cnt_loss
        loss_info["total_loss"] = total_loss.detach()
        loss_info = dict(sorted(loss_info.items()))  # sort by key for nicer printing

        return total_loss, loss_info