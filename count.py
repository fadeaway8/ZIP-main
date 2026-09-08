import torch
from torch import nn
import numpy as np
import os, json
from tqdm import tqdm
from argparse import ArgumentParser
from typing import Dict

import datasets


class SumPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: int):
        super(SumPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.sum_pool = nn.AvgPool2d(kernel_size, stride, divisor_override=1)

    def forward(self, x):
        return self.sum_pool(x)


def _update_dict(d: Dict, keys: np.ndarray, values: np.ndarray) -> Dict:
    keys = keys.tolist() if isinstance(keys, np.ndarray) else keys
    values = values.tolist() if isinstance(values, np.ndarray) else values
    for k, v in zip(keys, values):
        d[k] = d.get(k, 0) + v

    return d


def _get_counts(
    dataset_name: str,
    device: torch.device,
) -> None:
    filter_4 = SumPool2d(4, 1).to(device)
    filter_7 = SumPool2d(7, 1).to(device)
    filter_8 = SumPool2d(8, 1).to(device)
    filter_14 = SumPool2d(14, 1).to(device)
    filter_16 = SumPool2d(16, 1).to(device)
    filter_28 = SumPool2d(28, 1).to(device)
    filter_32 = SumPool2d(32, 1).to(device)
    filter_56 = SumPool2d(56, 1).to(device)
    filter_64 = SumPool2d(64, 1).to(device)
    counts_1, counts_4, counts_7, counts_8 = {}, {}, {}, {}
    counts_14, counts_16 = {}, {}
    counts_28, counts_32 = {}, {}
    counts_56, counts_64 = {}, {}

    max_counts_4 = {"max": 0., "name": None, "x": None, "y": None}
    max_counts_7 = {"max": 0., "name": None, "x": None, "y": None}
    max_counts_8 = {"max": 0., "name": None, "x": None, "y": None}
    max_counts_14 = {"max": 0., "name": None, "x": None, "y": None}
    max_counts_16 = {"max": 0., "name": None, "x": None, "y": None}
    max_counts_28 = {"max": 0., "name": None, "x": None, "y": None}
    max_counts_32 = {"max": 0., "name": None, "x": None, "y": None}
    max_counts_56 = {"max": 0., "name": None, "x": None, "y": None}
    max_counts_64 = {"max": 0., "name": None, "x": None, "y": None}

    counts_dir = os.path.join(os.getcwd(), "counts")
    os.makedirs(counts_dir, exist_ok=True)

    dataset = datasets.Crowd(dataset=dataset_name, split="train", transforms=None, return_filename=True)
    print(f"Counting {dataset_name} dataset")

    for i in tqdm(range(len(dataset))):
        _, _, density, img_name = dataset[i]
        density_np = density.cpu().numpy().astype(int)
        uniques_, counts_ = np.unique(density_np, return_counts=True)
        counts_1 = _update_dict(counts_1, uniques_, counts_)

        density = density.to(device)  # Add batch dimension
        window_4, window_7, window_8 = filter_4(density), filter_7(density), filter_8(density)
        window_14, window_16 = filter_14(density), filter_16(density)
        window_28, window_32 = filter_28(density), filter_32(density)
        window_56, window_64 = filter_56(density), filter_64(density)

        window_4, window_7, window_8 = torch.round(window_4).int(), torch.round(window_7).int(), torch.round(window_8).int()
        window_14, window_16 = torch.round(window_14).int(), torch.round(window_16).int()
        window_28, window_32 = torch.round(window_28).int(), torch.round(window_32).int()
        window_56, window_64 = torch.round(window_56).int(), torch.round(window_64).int()

        window_4, window_7, window_8 = torch.squeeze(window_4), torch.squeeze(window_7), torch.squeeze(window_8)
        window_14, window_16 = torch.squeeze(window_14), torch.squeeze(window_16)
        window_28, window_32 = torch.squeeze(window_28), torch.squeeze(window_32)
        window_56, window_64 = torch.squeeze(window_56), torch.squeeze(window_64)

        if window_4.max().item() > max_counts_4["max"]:
            max_counts_4["max"] = window_4.max().item()
            max_counts_4["name"] = img_name
            x, y = torch.where(window_4 == window_4.max())
            x, y = x[0].item(), y[0].item()
            max_counts_4["x"] = x
            max_counts_4["y"] = y
        
        if window_7.max().item() > max_counts_7["max"]:
            max_counts_7["max"] = window_7.max().item()
            max_counts_7["name"] = img_name
            x, y = torch.where(window_7 == window_7.max())
            x, y = x[0].item(), y[0].item()
            max_counts_7["x"] = x
            max_counts_7["y"] = y
        
        if window_8.max().item() > max_counts_8["max"]:
            max_counts_8["max"] = window_8.max().item()
            max_counts_8["name"] = img_name
            x, y = torch.where(window_8 == window_8.max())
            x, y = x[0].item(), y[0].item()
            max_counts_8["x"] = x
            max_counts_8["y"] = y
        
        if window_14.max().item() > max_counts_14["max"]:
            max_counts_14["max"] = window_14.max().item()
            max_counts_14["name"] = img_name
            x, y = torch.where(window_14 == window_14.max())
            x, y = x[0].item(), y[0].item()
            max_counts_14["x"] = x
            max_counts_14["y"] = y
        
        if window_16.max().item() > max_counts_16["max"]:
            max_counts_16["max"] = window_16.max().item()
            max_counts_16["name"] = img_name
            x, y = torch.where(window_16 == window_16.max())
            x, y = x[0].item(), y[0].item()
            max_counts_16["x"] = x
            max_counts_16["y"] = y
        
        if window_28.max().item() > max_counts_28["max"]:
            max_counts_28["max"] = window_28.max().item()
            max_counts_28["name"] = img_name
            x, y = torch.where(window_28 == window_28.max())
            x, y = x[0].item(), y[0].item()
            max_counts_28["x"] = x
            max_counts_28["y"] = y
        
        if window_32.max().item() > max_counts_32["max"]:
            max_counts_32["max"] = window_32.max().item()
            max_counts_32["name"] = img_name
            x, y = torch.where(window_32 == window_32.max())
            x, y = x[0].item(), y[0].item()
            max_counts_32["x"] = x
            max_counts_32["y"] = y
        
        if window_56.max().item() > max_counts_56["max"]:
            max_counts_56["max"] = window_56.max().item()
            max_counts_56["name"] = img_name
            x, y = torch.where(window_56 == window_56.max())
            x, y = x[0].item(), y[0].item()
            max_counts_56["x"] = x
            max_counts_56["y"] = y
        
        if window_64.max().item() > max_counts_64["max"]:
            max_counts_64["max"] = window_64.max().item()
            max_counts_64["name"] = img_name
            x, y = torch.where(window_64 == window_64.max())
            x, y = x[0].item(), y[0].item()
            max_counts_64["x"] = x
            max_counts_64["y"] = y

        window_4 = window_4.view(-1).cpu().numpy().astype(int)
        window_7 = window_7.view(-1).cpu().numpy().astype(int)
        window_8 = window_8.view(-1).cpu().numpy().astype(int)
        window_14 = window_14.view(-1).cpu().numpy().astype(int)
        window_16 = window_16.view(-1).cpu().numpy().astype(int)
        window_28 = window_28.view(-1).cpu().numpy().astype(int)
        window_32 = window_32.view(-1).cpu().numpy().astype(int)
        window_56 = window_56.view(-1).cpu().numpy().astype(int)
        window_64 = window_64.view(-1).cpu().numpy().astype(int)
        #.view(-1).cpu().numpy().astype(int)

        uniques_, counts_ = np.unique(window_4, return_counts=True)
        counts_4 = _update_dict(counts_4, uniques_, counts_)

        uniques_, counts_ = np.unique(window_7, return_counts=True)
        counts_7 = _update_dict(counts_7, uniques_, counts_)

        uniques_, counts_ = np.unique(window_8, return_counts=True)
        counts_8 = _update_dict(counts_8, uniques_, counts_)

        uniques_, counts_ = np.unique(window_14, return_counts=True)
        counts_14 = _update_dict(counts_14, uniques_, counts_)

        uniques_, counts_ = np.unique(window_16, return_counts=True)
        counts_16 = _update_dict(counts_16, uniques_, counts_)

        uniques_, counts_ = np.unique(window_28, return_counts=True)
        counts_28 = _update_dict(counts_28, uniques_, counts_)

        uniques_, counts_ = np.unique(window_32, return_counts=True)
        counts_32 = _update_dict(counts_32, uniques_, counts_)

        uniques_, counts_ = np.unique(window_56, return_counts=True)
        counts_56 = _update_dict(counts_56, uniques_, counts_)

        uniques_, counts_ = np.unique(window_64, return_counts=True)
        counts_64 = _update_dict(counts_64, uniques_, counts_)

    counts = {
        1: counts_1,
        4: counts_4,
        7: counts_7,
        8: counts_8,
        14: counts_14,
        16: counts_16,
        28: counts_28,
        32: counts_32,
        56: counts_56,
        64: counts_64
    }

    max_counts = {
        4: max_counts_4,
        7: max_counts_7,
        8: max_counts_8,
        14: max_counts_14,
        16: max_counts_16,
        28: max_counts_28,
        32: max_counts_32,
        56: max_counts_56,
        64: max_counts_64
    }

    with open(os.path.join(counts_dir, f"{dataset_name}.json"), "w") as f:
        json.dump(counts, f)
    
    with open(os.path.join(counts_dir, f"{dataset_name}_max.json"), "w") as f:
        json.dump(max_counts, f)


def parse_args():
    parser = ArgumentParser(description="Get local counts of the dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["nwpu", "ucf_qnrf", "shanghaitech_a", "shanghaitech_b"],
        required=True,
        help="The dataset to use."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="The device to use."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.dataset = datasets.standardize_dataset_name(args.dataset)
    args.device = torch.device(args.device)
    _get_counts(args.dataset, args.device)
