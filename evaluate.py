import torch
from torch.amp import autocast
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn, Tensor
from torch.utils.data import DataLoader
from typing import Tuple, Optional
from tqdm import tqdm
import numpy as np

from utils import sliding_window_predict, barrier, calculate_errors


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    sliding_window: bool,
    max_input_size: int = 4096,
    window_size: int = 224,
    stride: int = 224,
    max_num_windows: int = 64,
    device: torch.device = torch.device("cuda"),
    amp: bool = False,
    local_rank: int = 0,
    nprocs: int = 1,
    progress_bar: bool = True,
) -> Tuple[Tensor, Tensor]:
    ddp = nprocs > 1
    model = model.to(device)
    model.eval()
    pred_counts, gt_counts = [], []
    data_iter = tqdm(data_loader) if (local_rank == 0 and progress_bar) else data_loader

    for image, gt_points, _ in data_iter:
        image = image.to(device)
        image_height, image_width = image.shape[-2:]
        gt_counts.extend([len(p) for p in gt_points])

        # Resize image if it's smaller than the window size
        aspect_ratio = image_width / image_height
        if image_height < window_size:
            new_height = window_size
            new_width = int(new_height * aspect_ratio)
            image = F.interpolate(image, size=(new_height, new_width), mode="bicubic", align_corners=False)
            image_height, image_width = new_height, new_width
        if image_width < window_size:
            new_width = window_size
            new_height = int(new_width / aspect_ratio)
            image = F.interpolate(image, size=(new_height, new_width), mode="bicubic", align_corners=False)
            image_height, image_width = new_height, new_width

        with torch.set_grad_enabled(False), autocast(device_type="cuda", enabled=amp):
            if sliding_window or (image_height * image_width) > max_input_size ** 2:
                pred_den_maps = sliding_window_predict(model, image, window_size, stride, max_num_windows)
            else:
                pred_den_maps = model(image)

            pred_counts.extend(pred_den_maps.sum(dim=(-1, -2, -3)).cpu().numpy().tolist())
    
    barrier(ddp)
    assert len(pred_counts) == len(gt_counts), f"Length of predictions and ground truths should be equal, but got {len(pred_counts)} and {len(gt_counts)}"

    if ddp:
        pred_counts, gt_counts = torch.tensor(pred_counts, device=device), torch.tensor(gt_counts, device=device)
        # Pad `pred_counts` and `gt_counts` to the same length across all processes.
        local_length = torch.tensor([len(pred_counts)], device=device)
        lengths = [torch.zeros_like(local_length) for _ in range(nprocs)]
        dist.all_gather(lengths, local_length)
        max_length = max([l.item() for l in lengths])
        padded_pred_counts, padded_gt_counts = torch.full((max_length,), float("nan"), device=device), torch.full((max_length,), float("nan"), device=device)
        padded_pred_counts[:len(pred_counts)], padded_gt_counts[:len(gt_counts)] = pred_counts, gt_counts
        gathered_pred_counts, gathered_gt_counts = [torch.zeros_like(padded_pred_counts) for _ in range(nprocs)], [torch.zeros_like(padded_gt_counts) for _ in range(nprocs)]
        dist.all_gather(gathered_pred_counts, padded_pred_counts)
        dist.all_gather(gathered_gt_counts, padded_gt_counts)
        # Concatenate predictions and ground truths from all processes and remove padding (nan values).
        pred_counts, gt_counts = torch.cat(gathered_pred_counts).cpu(), torch.cat(gathered_gt_counts).cpu()
        pred_counts, gt_counts = pred_counts[~torch.isnan(pred_counts)], gt_counts[~torch.isnan(gt_counts)]
        pred_counts, gt_counts = pred_counts.numpy(), gt_counts.numpy()

    else:
        pred_counts, gt_counts = np.array(pred_counts), np.array(gt_counts)

    torch.cuda.empty_cache()
    return calculate_errors(pred_counts, gt_counts)
