import torch
from torch import nn, Tensor
from torch.amp import autocast
from typing import List, Tuple, Dict

from .bregman_pytorch import sinkhorn
from .utils import _reshape_density

EPS = 1e-8


class OTLoss(nn.Module):
    def __init__(
        self,
        input_size: int,
        block_size: int,
        numItermax: int = 100,
        regularization: float = 10.0
    ) -> None:
        super().__init__()
        assert input_size % block_size == 0

        self.input_size = input_size
        self.block_size = block_size
        self.num_blocks_h = input_size // block_size
        self.num_blocks_w = input_size // block_size
        self.numItermax = numItermax
        self.regularization = regularization

        # coordinate is same to image space, set to constant since crop size is same
        self.coords_h = torch.arange(0, input_size, step=block_size, dtype=torch.float32) + block_size / 2
        self.coords_w = torch.arange(0, input_size, step=block_size, dtype=torch.float32) + block_size / 2
        self.coords_h, self.coords_w = self.coords_h.unsqueeze(0), self.coords_w.unsqueeze(0)  # [1, #coordinates]

    def set_numItermax(self, numItermax: int) -> None:
        self.numItermax = numItermax

    @autocast(device_type="cuda", enabled=True, dtype=torch.float32)  # avoid numerical instability
    def forward(self, pred_den_map: Tensor, pred_den_map_normed: Tensor, gt_points: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        assert pred_den_map.shape[1:] == pred_den_map_normed.shape[1:] == (1, self.num_blocks_h, self.num_blocks_w), f"Expected pred_den_map to have shape (B, 1, {self.num_blocks_h}, {self.num_blocks_w}), but got {pred_den_map.shape} and {pred_den_map_normed.shape}"
        assert len(gt_points) == pred_den_map.shape[0] == pred_den_map_normed.shape[0], f"Expected gt_points to have length {pred_den_map_normed.shape[0]}, but got {len(gt_points)}"
        device = pred_den_map.device

        loss = torch.zeros(1, device=device)
        ot_obj_values = torch.zeros(1, device=device)
        w_dist = torch.zeros(1, device=device)  # Wasserstein distance
        coords_h, coords_w = self.coords_h.to(device), self.coords_w.to(device)  # [1, #coordinates]
        for idx, points in enumerate(gt_points):
            if len(points) > 0:
                # compute l2 square distance, it should be source target distance. [#gt, #coordinates * #coordinates]
                x, y = points[:, 0].unsqueeze(1), points[:, 1].unsqueeze(1)  # [#gt, 1]
                x_dist = -2 * torch.matmul(x, coords_w) + x * x + coords_w * coords_w  # [#gt, #coordinates]
                y_dist = -2 * torch.matmul(y, coords_h) + y * y + coords_h * coords_h  # [#gt, #coordinates]
                dist = x_dist.unsqueeze(1) + y_dist.unsqueeze(2)
                dist = dist.view((dist.shape[0], -1)) # size of [#gt, #coordinates * #coordinates]

                source_prob = pred_den_map_normed[idx].view(-1).detach()
                target_prob = (torch.ones(len(points)) / len(points)).to(device)
                # use sinkhorn to solve OT, compute optimal beta.
                P, log = sinkhorn(
                    a=target_prob,
                    b=source_prob,
                    C=dist,
                    reg=self.regularization,
                    maxIter=self.numItermax,
                    log=True
                )
                beta = log["beta"] # size is the same as source_prob: [#coordinates * #coordinates]
                w_dist += (dist * P).sum()
                ot_obj_values += (pred_den_map_normed[idx] * beta.view(1, self.num_blocks_h, self.num_blocks_w)).sum()
                # compute the gradient of OT loss to predicted density (pred_den_map).
                # im_grad = beta / source_count - < beta, source_density> / (source_count)^2
                source_density = pred_den_map[idx].view(-1).detach()
                source_count = source_density.sum()
                gradient_1 = (source_count) / (source_count * source_count+ EPS) * beta # size of [#coordinates * #coordinates]
                gradient_2 = (source_density * beta).sum() / (source_count * source_count + EPS) # size of 1
                gradient = gradient_1 - gradient_2
                gradient = gradient.detach().view(1, self.num_blocks_h, self.num_blocks_w)
                # Define loss = <im_grad, predicted density>. The gradient of loss w.r.t predicted density is im_grad.
                loss += torch.sum(pred_den_map[idx] * gradient)

        return loss, w_dist, ot_obj_values


class DMLoss(nn.Module):
    def __init__(
        self,
        input_size: int,
        block_size: int,
        numItermax: int = 100,
        regularization: float = 10.0,
        weight_ot: float = 0.1,
        weight_tv: float = 0.01,
        weight_cnt: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.block_size = block_size
        self.weight_ot = weight_ot
        self.weight_tv = weight_tv
        self.weight_cnt = weight_cnt

        self.ot_loss = OTLoss(
            input_size=self.input_size,
            block_size=self.block_size,
            numItermax=numItermax,
            regularization=regularization,
        )
        self.tv_loss = nn.L1Loss(reduction="none")
        self.cnt_loss = nn.L1Loss(reduction="mean")
        self.weight_ot = weight_ot
        self.weight_tv = weight_tv

    @autocast(device_type="cuda", enabled=True, dtype=torch.float32)  # avoid numerical instability
    def forward(self, pred_den_map: Tensor, gt_den_map: Tensor, gt_points: List[Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        gt_den_map = _reshape_density(gt_den_map, block_size=self.ot_loss.block_size) if gt_den_map.shape[-2:] != pred_den_map.shape[-2:] else gt_den_map
        assert pred_den_map.shape == gt_den_map.shape, f"Expected pred_den_map and gt_den_map to have the same shape, got {pred_den_map.shape} and {gt_den_map.shape}"

        pred_cnt = pred_den_map.view(pred_den_map.shape[0], -1).sum(dim=1)
        pred_den_map_normed = pred_den_map / (pred_cnt.view(-1, 1, 1, 1) + EPS)
        gt_cnt = torch.tensor([len(p) for p in gt_points], dtype=torch.float32).to(pred_den_map.device)
        gt_den_map_normed = gt_den_map / (gt_cnt.view(-1, 1, 1, 1) + EPS)

        ot_loss, w_dist, _ = self.ot_loss(pred_den_map, pred_den_map_normed, gt_points)

        tv_loss = (self.tv_loss(pred_den_map_normed, gt_den_map_normed).sum(dim=(1, 2, 3)) * gt_cnt).mean() if self.weight_tv > 0 else 0

        cnt_loss = self.cnt_loss(pred_cnt, gt_cnt) if self.weight_cnt > 0 else 0

        loss = ot_loss * self.weight_ot + tv_loss * self.weight_tv + cnt_loss * self.weight_cnt

        loss_info = {
            "ot_loss": ot_loss.detach(),
            "dm_loss": loss.detach(),
            "w_dist": w_dist.detach(),
        }
        if self.weight_tv > 0:
            loss_info["tv_loss"] = tv_loss.detach()
        if self.weight_cnt > 0:
            loss_info["cnt_loss"] = cnt_loss.detach()

        return loss, loss_info
