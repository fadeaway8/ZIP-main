import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .utils import _reshape_density


def ramp_weight(epoch: int, max_weight: float, warmup_epochs: int, ramp_epochs: int) -> float:
    if max_weight <= 0:
        return 0.0
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return float(max_weight)

    progress = (epoch - warmup_epochs) / float(ramp_epochs)
    progress = max(0.0, min(1.0, progress))
    return float(max_weight) * progress


def gt_hdba_candidate_block_mask(
    gt_den_map: Tensor,
    gt_points: List[Tensor],
    input_size: int,
    block_size: int,
    out_h: int,
    out_w: int,
    img_count_thr: float,
    top_ratio: float = 0.2,
    block_count_thr: float = 0.0,
) -> Tensor:
    if gt_den_map.shape[-2:] != (out_h, out_w):
        assert gt_den_map.shape[-2:] == (input_size, input_size)
        gt_den_map = _reshape_density(gt_den_map, block_size=block_size)

    img_counts = torch.tensor(
        [len(points) for points in gt_points],
        dtype=gt_den_map.dtype,
        device=gt_den_map.device,
    ).view(-1, 1, 1, 1)
    high_img_mask = img_counts >= img_count_thr

    block_count = gt_den_map
    positive_mask = block_count > 0
    candidate_mask = torch.zeros_like(positive_mask, dtype=torch.bool)

    if block_count_thr > 0:
        candidate_mask = positive_mask & (block_count >= block_count_thr)
    else:
        top_ratio = max(0.0, min(1.0, float(top_ratio)))
        if top_ratio > 0:
            for idx in range(block_count.shape[0]):
                positive_counts = block_count[idx][positive_mask[idx]]
                if positive_counts.numel() == 0:
                    continue
                k = max(1, math.ceil(positive_counts.numel() * top_ratio))
                threshold = positive_counts.topk(k).values[-1]
                candidate_mask[idx] = positive_mask[idx] & (block_count[idx] >= threshold)

    return candidate_mask & high_img_mask


def zip_hdba(
    model: nn.Module,
    lambda_logit: Tensor,
    pi_logit: Optional[Tensor] = None,
    block_mask: Optional[Tensor] = None,
    high_bin_min_count: float = 10.0,
    margin: float = 0.02,
    pi_nonzero_thr: float = 0.4,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    m = model.module if hasattr(model, "module") else model
    prob = lambda_logit.softmax(dim=1)
    centers = m.bin_centers.view(-1)[1:].to(device=prob.device, dtype=prob.dtype)
    hi = centers >= high_bin_min_count

    if hi.sum() == 0:
        z = prob.mean() * 0.0
        return z, {
            "hdba_num_blocks": z.detach(),
            "lambda_high_bin_mass_pos": z.detach(),
            "lambda_expected_bin_index_pos": z.detach(),
        }

    high_mass = prob[:, hi].sum(dim=1, keepdim=True)
    bin_indices = torch.arange(
        1,
        prob.size(1) + 1,
        device=prob.device,
        dtype=prob.dtype,
    ).view(1, -1, 1, 1)
    exp_idx = (prob * bin_indices).sum(dim=1, keepdim=True)

    if block_mask is None:
        mask = torch.ones_like(high_mass, dtype=torch.bool)
    else:
        mask = block_mask.to(device=prob.device).bool()
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    if pi_logit is not None:
        p_nonzero = pi_logit.softmax(dim=1)[:, 1:2]
        mask = mask & (p_nonzero > pi_nonzero_thr)

    if mask.sum() == 0:
        z = high_mass.mean() * 0.0
        return z, {
            "hdba_num_blocks": z.detach(),
            "lambda_high_bin_mass_pos": z.detach(),
            "lambda_expected_bin_index_pos": z.detach(),
        }

    loss = F.relu(margin - high_mass[mask]).mean()
    return loss, {
        "hdba_num_blocks": mask.sum().to(dtype=high_mass.dtype).detach(),
        "lambda_high_bin_mass_pos": high_mass[mask].mean().detach(),
        "lambda_expected_bin_index_pos": exp_idx[mask].mean().detach(),
    }
