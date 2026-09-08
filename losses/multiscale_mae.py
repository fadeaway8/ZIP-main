from torch import nn, Tensor
import math
from typing import List, Optional, Dict, Tuple


class MultiscaleMAE(nn.Module):
    def __init__(
        self,
        scales: List[int] = [1, 2, 4],
        min_scale_weight: float = 0.0,
        max_scale_weight: float = 1.0,
        alpha: float = 0.5,
        weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        assert isinstance(scales, (list, tuple)) and len(scales) > 0 and all(isinstance(s, int) and s > 0 for s in scales), f"Expected scales to be a list of positive integers, got {scales}"
        assert max_scale_weight >= min_scale_weight >= 0, f"Expected max_scale_weight to be greater than or equal to min_scale_weight, got {min_scale_weight} and {max_scale_weight}"
        assert 1 > alpha > 0, f"Expected alpha to be between 0 and 1, got {alpha}"
        self.min_scale_weight, self.max_scale_weight = min_scale_weight, max_scale_weight

        scales = sorted(scales)  # sort scales in ascending order so that the last one is the largest
        weights = [min_scale_weight + (max_scale_weight - min_scale_weight) * alpha ** (math.log2(scales[-1] / s)) for s in scales] if weights is None else weights  # e.g., [1, 2, 4, 8] -> [0.125, 0.25, 0.5, 1]

        assert len(scales) == len(weights), f"Expected scales and weights to have the same length, got {len(scales)} and {len(weights)}"
        self.scales, self.weights = scales, weights

        for idx in range(len(scales)):
            setattr(self, f"pool_{scales[idx]}", nn.AvgPool2d(kernel_size=scales[idx], stride=scales[idx]) if scales[idx] > 1 else nn.Identity())
            setattr(self, f"weight_{scales[idx]}", weights[idx])
            setattr(self, f"mae_loss_fn_{scales[idx]}", nn.L1Loss(reduction="none"))

    def forward(
        self,
        pred_den_map: Tensor,
        gt_den_map: Tensor,
    ) -> Tuple[Tensor, Dict]:
        assert len(pred_den_map.shape) == 4, f"Expected pred_den_map to have 4 dimensions, got {len(pred_den_map.shape)}"
        assert len(gt_den_map.shape) == 4, f"Expected gt_den_map to have 4 dimensions, got {len(gt_den_map.shape)}"
        assert pred_den_map.shape[1] == gt_den_map.shape[1] == 1, f"Expected pred_den_map and gt_den_map to have 1 channel, got {pred_den_map.shape[1]} and {gt_den_map.shape[1]}"
        assert pred_den_map.shape == gt_den_map.shape, f"Expected pred_den_map and gt_den_map to have the same shape, got {pred_den_map.shape} and {gt_den_map.shape}"
        
        loss, loss_info = 0, {}
        for idx in range(len(self.scales)):
            pool = getattr(self, f"pool_{self.scales[idx]}")
            weight = getattr(self, f"weight_{self.scales[idx]}")
            loss_fn = getattr(self, f"mae_loss_fn_{self.scales[idx]}")

            pred_den_map_pool = pool(pred_den_map)
            gt_den_map_pool = pool(gt_den_map)

            mae_loss_scale = loss_fn(pred_den_map_pool, gt_den_map_pool).sum(dim=(-1, -2)).mean()
            loss += weight * mae_loss_scale
            loss_info[f"mae_loss_{self.scales[idx]}"] = mae_loss_scale.detach()
            
        return loss, loss_info
