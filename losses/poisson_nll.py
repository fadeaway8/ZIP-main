import torch
from torch import nn, Tensor
from .utils import _reshape_density


EPS = 1e-8


class PoissonNLL(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        assert reduction in ["none", "mean", "sum"], f"Expected reduction to be one of ['none', 'mean', 'sum'], got {reduction}."
        self.reduction = reduction

    def forward(self, pred_den_map: Tensor, gt_den_map: Tensor) -> Tensor:
        """
        Args:
            pred_den_map: predicted λ map, shape (B, 1, H, W)
            gt_den_map: ground truth density map, shape (B, 1, H, W)
        Returns:
            Poisson loss
        """
        assert len(pred_den_map.shape) == 4, f"Expected pred_den_map to have 4 dimensions, got {len(pred_den_map.shape)}"
        assert len(gt_den_map.shape) == 4, f"Expected gt_den_map to have 4 dimensions, got {len(gt_den_map.shape)}"
        assert pred_den_map.shape[1] == gt_den_map.shape[1] == 1, f"Expected pred_den_map and gt_den_map to have 1 channel, got {pred_den_map.shape[1]} and {gt_den_map.shape[1]}"
        if gt_den_map.shape != pred_den_map.shape:
            gt_h, gt_w = gt_den_map.shape[-2], gt_den_map.shape[-1]
            pred_h, pred_w = pred_den_map.shape[-2], pred_den_map.shape[-1]
            assert gt_h % pred_h == 0 and gt_w % pred_w == 0 and gt_h // pred_h == gt_w // pred_w, f"Expected the spatial dimension of gt_den_map to be a multiple of that of pred_den_map, got {gt_den_map.shape} and {pred_den_map.shape}"
            gt_den_map = _reshape_density(gt_den_map, block_size=gt_h // pred_h)
        
        assert gt_den_map.shape == pred_den_map.shape, f"Expected gt_den_map and pred_den_map to have the same shape, got {gt_den_map.shape} and {pred_den_map.shape}"

        gt_den_map = gt_den_map.to(pred_den_map.device)

        loss = (pred_den_map - gt_den_map * torch.log(pred_den_map + EPS)).sum(dim=(-1, -2))  # sum over H and W

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        
        return loss, {"pnll": loss.detach()}
