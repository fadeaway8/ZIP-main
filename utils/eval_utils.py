import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Union


def calculate_errors(pred_counts: np.ndarray, gt_counts: np.ndarray) -> Dict[str, float]:
    assert isinstance(pred_counts, np.ndarray), f"Expected numpy.ndarray, got {type(pred_counts)}"
    assert isinstance(gt_counts, np.ndarray), f"Expected numpy.ndarray, got {type(gt_counts)}"
    assert len(pred_counts) == len(gt_counts), f"Length of predictions and ground truths should be equal, but got {len(pred_counts)} and {len(gt_counts)}"
    indices = gt_counts > 0
    errors = {
        "mae": np.mean(np.abs(pred_counts - gt_counts)),
        "rmse": np.sqrt(np.mean((pred_counts - gt_counts) ** 2)),
        "nae": np.mean(np.abs(pred_counts[indices] - gt_counts[indices]) / gt_counts[indices])
    }
    return errors


def resize_density_map(x: Tensor, size: Tuple[int, int]) -> Tensor:
    x_sum = torch.sum(x, dim=(-1, -2))
    x = F.interpolate(x, size=size, mode="bilinear")
    scale_factor = torch.nan_to_num(torch.sum(x, dim=(-1, -2)) / x_sum, nan=0.0, posinf=0.0, neginf=0.0)
    return x * scale_factor


def sliding_window_predict(
    model: nn.Module,
    image: Tensor,
    window_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    max_num_windows: int,
) -> Tensor:
    """
    Generate the density map for an image using the sliding window method. Overlapping regions will be averaged.

    Args:
        model (nn.Module): The model to use.
        image (Tensor): The image (1, c, h, w) to generate the density map for. The batch size must be 1 due to varying image sizes.
        window_size (Union[int, Tuple[int, int]]): The size of the window.
        stride (Union[int, Tuple[int, int]]): The step size of the window.
    """
    assert len(image.shape) == 4, f"Image must be a 4D tensor (1, c, h, w), got {image.shape}"
    window_size = (int(window_size), int(window_size)) if isinstance(window_size, (int, float)) else window_size
    stride = (int(stride), int(stride)) if isinstance(stride, (int, float)) else stride
    window_size = tuple(window_size)
    stride = tuple(stride)
    assert isinstance(window_size, tuple) and len(window_size) == 2 and window_size[0] > 0 and window_size[1] > 0, f"Window size must be a positive integer tuple (h, w), got {window_size}"
    assert isinstance(stride, tuple) and len(stride) == 2 and stride[0] > 0 and stride[1] > 0, f"Stride must be a positive integer tuple (h, w), got {stride}"
    assert stride[0] <= window_size[0] and stride[1] <= window_size[1], f"Stride must be smaller than window size, got {stride} and {window_size}"

    image_height, image_width = image.shape[-2:]
    window_height, window_width = window_size
    assert image_height >= window_height and image_width >= window_width, f"Image size must be larger than window size, got image size {image.shape} and window size {window_size}"
    stride_height, stride_width = stride

    num_rows = int(np.ceil((image_height - window_height) / stride_height) + 1)
    num_cols = int(np.ceil((image_width - window_width) / stride_width) + 1)

    if hasattr(model, "block_size"):
        block_size = model.block_size
    elif hasattr(model, "module") and hasattr(model.module, "block_size"):
        block_size = model.module.block_size
    else:
        raise ValueError("Model must have block_size attribute")
    assert window_height % block_size == 0 and window_width % block_size == 0, f"Window size must be divisible by block size, got {window_size} and {block_size}"

    windows = []
    for i in range(num_rows):
        for j in range(num_cols):
            x_start, y_start = i * stride_height, j * stride_width
            x_end, y_end = x_start + window_height, y_start + window_width
            if x_end > image_height:
                x_start, x_end = image_height - window_height, image_height
            if y_end > image_width:
                y_start, y_end = image_width - window_width, image_width

            window = image[:, :, x_start:x_end, y_start:y_end]
            windows.append(window)

    windows = torch.cat(windows, dim=0).to(image.device)  # batched windows, shape: (num_windows, c, h, w)
    
    model.eval()
    preds = []
    for i in range(0, len(windows), max_num_windows):
        with torch.no_grad():
            preds_ = model(windows[i: min(i + max_num_windows, len(windows))])
        preds.append(preds_.cpu().numpy())
    preds = np.concatenate(preds, axis=0)  # shape: (num_windows, 1, h // block_size, w // block_size)

    # assemble the density map
    pred_map = np.zeros((preds.shape[1], image_height // block_size, image_width // block_size), dtype=np.float32)
    count_map = np.zeros((preds.shape[1], image_height // block_size, image_width // block_size), dtype=np.float32)
    idx = 0
    for i in range(num_rows):
        for j in range(num_cols):
            x_start, y_start = i * stride_height, j * stride_width
            x_end, y_end = x_start + window_height, y_start + window_width
            if x_end > image_height:
                x_start, x_end = image_height - window_height, image_height
            if y_end > image_width:
                y_start, y_end = image_width - window_width, image_width

            pred_map[:, (x_start // block_size): (x_end // block_size), (y_start // block_size): (y_end // block_size)] += preds[idx, :, :, :]
            count_map[:, (x_start // block_size): (x_end // block_size), (y_start // block_size): (y_end // block_size)] += 1.
            idx += 1

    pred_map /= count_map  # average the overlapping regions
    preds = torch.tensor(pred_map).unsqueeze(0)  # shape: (1, 1, h // block_size, w // block_size)
    return preds