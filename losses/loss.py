import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union

from .dm_loss import DMLoss
from .multiscale_mae import MultiscaleMAE
from .poisson_nll import PoissonNLL
from .zero_inflated_poisson_nll import ZIPoissonNLL, ZICrossEntropy
from .utils import _reshape_density, _bin_count


EPS = 1e-8


class QuadLoss(nn.Module):
    def __init__(
        self,
        input_size: int,
        block_size: int,
        bins: List[Tuple[float, float]],
        reg_loss: str = "zipnll",
        aux_loss: str = "none",
        weight_cls: float = 1.0,
        weight_reg: float = 1.0,
        weight_aux: Optional[float] = None,
        numItermax: Optional[int] = 100,
        regularization: Optional[int] = 10.0,
        scales: Optional[List[int]] = [[1, 2, 4]],
        min_scale_weight: Optional[float] = 0.0,
        max_scale_weight: Optional[float] = 1.0,
        alpha: Optional[float] = 0.5,
    ) -> None:
        super().__init__()
        assert input_size % block_size == 0, f"Expected input_size to be divisible by block_size, got {input_size} and {block_size}"
        assert len(bins) >= 2, f"Expected bins to have at least 2 elements, got {len(bins)}"
        assert all([len(b) == 2 for b in bins]), f"Expected all bins to be of length 2, got {bins}"
        bins = [(float(low), float(high)) for low, high in bins]
        assert all([b[0] <= b[1] for b in bins]), f"Expected each bin to have bin[0] <= bin[1], got {bins}"
        assert reg_loss in ["zipnll", "pnll", "dm", "msmae", "mae", "mse"], f"Expected reg_loss to be one of ['zipnll', 'pnll', 'dm', 'msmae', 'mae', 'mse'], got {reg_loss}"
        assert aux_loss in ["zipnll", "pnll", "dm", "msmae", "mae", "mse", "none"], f"Expected aux_loss to be one of ['zipnll', 'pnll', 'dm', 'msmae', 'mae', 'mse', 'none'], got {aux_loss}"

        assert weight_cls >= 0, f"Expected weight_cls to be non-negative, got {weight_cls}"
        assert weight_reg >= 0, f"Expected weight_reg to be non-negative, got {weight_reg}"
        assert not (weight_cls == 0 and weight_reg == 0), "Expected at least one of weight_cls and weight_reg to be non-zero"
        weight_aux = 0 if aux_loss == "none" or weight_aux is None else weight_aux
        assert weight_aux >= 0, f"Expected weight_aux to be non-negative, got {weight_aux}"

        self.input_size = input_size
        self.block_size = block_size
        self.bins = bins
        self.reg_loss = reg_loss
        self.aux_loss = aux_loss
        self.weight_cls = weight_cls
        self.weight_reg = weight_reg
        self.weight_aux = weight_aux

        self.num_bins = len(bins)
        self.num_blocks_h = input_size // block_size
        self.num_blocks_w = input_size // block_size

        if reg_loss == "zipnll":
            self.cls_loss = "zice"
            self.cls_loss_fn = ZICrossEntropy(bins=bins, reduction="mean")
            self.reg_loss_fn = ZIPoissonNLL(reduction="mean")
        else:
            self.cls_loss = "ce"
            self.cls_loss_fn = nn.CrossEntropyLoss(reduction="none")
            if reg_loss == "pnll":
                self.reg_loss_fn = PoissonNLL(reduction="mean")
            elif reg_loss == "dm":
                assert numItermax is not None and numItermax > 0, f"Expected numItermax to be a positive integer, got {numItermax}"
                assert regularization is not None and regularization > 0, f"Expected regularization to be a positive float, got {regularization}"
                self.reg_loss_fn = DMLoss(
                    input_size=input_size,
                    block_size=block_size,
                    numItermax=numItermax,
                    regularization=regularization,
                    weight_ot=0.1,
                    weight_tv=0.01,
                    weight_cnt=0,  # count loss will be calculated separately in this module.
                )
            elif reg_loss == "msmae":
                assert isinstance(scales, (list, tuple)) and len(scales) > 0 and all(isinstance(s, int) and s > 0 for s in scales), f"Expected scales to be a list of positive integers, got {scales}"
                assert max_scale_weight >= min_scale_weight >= 0, f"Expected max_scale_weight to be greater than or equal to min_scale_weight, got {min_scale_weight} and {max_scale_weight}"
                assert 1 > alpha > 0, f"Expected alpha to be between 0 and 1, got {alpha}"
                self.reg_loss_fn = MultiscaleMAE(
                    scales=sorted(scales),
                    min_scale_weight=min_scale_weight,
                    max_scale_weight=max_scale_weight,
                    alpha=alpha,
                )
            elif reg_loss == "mae":
                self.reg_loss_fn = nn.L1Loss(reduction="none")
            elif reg_loss == "mse":
                self.reg_loss_fn = nn.MSELoss(reduction="none")
            else:  # reg_loss == "none"
                self.reg_loss_fn = None

        if aux_loss == "zipnll":
            self.aux_loss_fn = ZIPoissonNLL(reduction="mean")
        elif aux_loss == "pnll":
            self.aux_loss_fn = PoissonNLL(reduction="mean")
        elif aux_loss == "dm":
            assert numItermax is not None and numItermax > 0, f"Expected numItermax to be a positive integer, got {numItermax}"
            assert regularization is not None and regularization > 0, f"Expected regularization to be a positive float, got {regularization}"
            self.aux_loss_fn = DMLoss(
                input_size=input_size,
                block_size=block_size,
                numItermax=numItermax,
                regularization=regularization,
                weight_ot=0.1,
                weight_tv=0.01,
                weight_cnt=0,  # count loss will be calculated separately in this module.
            )
        elif aux_loss == "msmae":
            assert isinstance(scales, (list, tuple)) and len(scales) > 0 and all(isinstance(s, int) and s > 0 for s in scales), f"Expected scales to be a list of positive integers, got {scales}"
            assert max_scale_weight >= min_scale_weight >= 0, f"Expected max_scale_weight to be greater than or equal to min_scale_weight, got {min_scale_weight} and {max_scale_weight}"
            assert 1 > alpha > 0, f"Expected alpha to be between 0 and 1, got {alpha}"
            self.aux_loss_fn = MultiscaleMAE(
                scales=sorted(scales),
                min_scale_weight=min_scale_weight,
                max_scale_weight=max_scale_weight,
                alpha=alpha,
            )
        elif aux_loss == "mae":
            self.aux_loss_fn = nn.L1Loss(reduction="none")
        elif aux_loss == "mse":
            self.aux_loss_fn = nn.MSELoss(reduction="none")
        else:  # aux_loss == "none"
            self.aux_loss_fn = None

        self.cnt_loss_fn = nn.L1Loss(reduction="mean")

    def forward(
        self,
        pred_logit_map: Tensor,
        pred_den_map: Tensor,
        gt_den_map: Tensor,
        gt_points: List[Tensor],
        pred_logit_pi_map: Optional[Tensor] = None,
        pred_lambda_map: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        B = pred_den_map.shape[0]
        assert pred_logit_map.shape[-2:] == (self.num_blocks_h, self.num_blocks_w), f"Expected pred_logit_map to have the spatial dimension of {self.num_blocks_h}x{self.num_blocks_w}, got {pred_logit_map.shape}"
        if gt_den_map.shape[-2:] != (self.num_blocks_h, self.num_blocks_w):
            assert gt_den_map.shape[-2:] == (self.input_size, self.input_size), f"Expected gt_den_map to have shape {B, 1, self.input_size, self.input_size}, got {gt_den_map.shape}"
            gt_den_map = _reshape_density(gt_den_map, block_size=self.block_size)
        assert pred_den_map.shape == gt_den_map.shape == (B, 1, self.num_blocks_h, self.num_blocks_w), f"Expected pred_den_map and gt_den_map to have shape (B, 1, H, W), got {pred_den_map.shape} and {gt_den_map.shape}"
        assert len(gt_points) == B, f"Expected gt_points to have length B, got {len(gt_points)}"

        if self.reg_loss == "zipnll" or self.aux_loss == "zipnll":
            assert pred_logit_pi_map is not None and pred_logit_pi_map.shape == (B, 2, self.num_blocks_h, self.num_blocks_w), f"Expected pred_logit_pi_map to have shape {B, 2, self.num_blocks_h, self.num_blocks_w}, got {pred_logit_pi_map.shape}"
            assert pred_lambda_map is not None and pred_lambda_map.shape == (B, 1, self.num_blocks_h, self.num_blocks_w), f"Expected pred_lambda_map to have shape {B, 1, self.num_blocks_h, self.num_blocks_w}, got {pred_lambda_map.shape}"
        
        loss_info = {}
        if self.weight_cls > 0:
            gt_class_map = _bin_count(gt_den_map, bins=self.bins)
            if self.cls_loss == "ce":
                cls_loss = self.cls_loss_fn(pred_logit_map, gt_class_map).sum(dim=(-1, -2)).mean()
                loss_info["cls_ce_loss"] = cls_loss.detach()
            else:  # cls_loss == "zice"
                cls_loss, cls_loss_info = self.cls_loss_fn(pred_logit_map, gt_den_map)
                loss_info.update(cls_loss_info)
        else:
            cls_loss = 0
        
        if self.weight_reg > 0:
            if self.reg_loss == "zipnll":
                reg_loss, reg_loss_info = self.reg_loss_fn(pred_logit_pi_map, pred_lambda_map, gt_den_map)
            elif self.reg_loss == "dm":
                reg_loss, reg_loss_info = self.reg_loss_fn(pred_den_map, gt_den_map, gt_points)
            elif self.reg_loss in ["pnll", "msmae"]:
                reg_loss, reg_loss_info = self.reg_loss_fn(pred_den_map, gt_den_map)
            else:  # reg_loss in ["mae", "mse"]
                reg_loss = self.reg_loss_fn(pred_den_map, gt_den_map).sum(dim=(-1, -2)).mean()
                reg_loss_info = {f"{self.reg_loss}": reg_loss.detach()}
            reg_loss_info = {f"reg_{k}": v for k, v in reg_loss_info.items()}
            loss_info.update(reg_loss_info)
        else:
            reg_loss = 0
        
        if self.weight_aux > 0:
            if self.aux_loss == "zipnll":
                aux_loss, aux_loss_info = self.aux_loss_fn(pred_logit_pi_map, pred_lambda_map, gt_den_map)
            elif self.aux_loss in ["pnll", "msmae"]:
                aux_loss, aux_loss_info = self.aux_loss_fn(pred_den_map, gt_den_map)
            elif self.aux_loss == "dm":
                aux_loss, aux_loss_info = self.aux_loss_fn(pred_den_map, gt_den_map, gt_points)
            else:
                aux_loss = self.aux_loss_fn(pred_den_map, gt_den_map).sum(dim=(-1, -2)).mean()
                aux_loss_info = {f"{self.aux_loss}": aux_loss.detach()}
            aux_loss_info = {f"aux_{k}": v for k, v in aux_loss_info.items()}
            loss_info.update(aux_loss_info)
        else:
            aux_loss = 0
        
        gt_cnt = torch.tensor([len(p) for p in gt_points], dtype=torch.float32, device=pred_den_map.device)
        cnt_loss = self.cnt_loss_fn(pred_den_map.sum(dim=(1, 2, 3)), gt_cnt)
        loss_info["cnt_loss"] = cnt_loss.detach()

        total_loss = self.weight_cls * cls_loss + self.weight_reg * reg_loss + self.weight_aux * aux_loss + cnt_loss
        return total_loss, loss_info
    