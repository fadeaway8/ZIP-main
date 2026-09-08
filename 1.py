import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets import Crowd, collate_fn, standardize_dataset_name
from models import get_model


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def zip_forward_maps(model, image: torch.Tensor, high_bin_min_count: float = 10.0) -> Dict[str, torch.Tensor]:
    """
    返回 ZIP 内部图：
    den_map: 最终预测 Y* = P(nonzero) * lambda
    lambda_map: 正计数分支的期望 lambda
    pi_zero_map: 结构零概率 pi = P(zero)
    lambda_high_mass_map: lambda 分支高密 bin 概率质量
    lambda_expected_bin_index_map: lambda 分支期望 bin index
    """
    m = unwrap_model(model)
    assert getattr(m, "zero_inflated", False), "该脚本只适用于 zero-inflated ZIP 模型。"
    assert hasattr(m, "pi_head") and hasattr(m, "lambda_head"), "模型没有暴露 ZIP 的 pi/lambda head。"

    image_feats = m.backbone(image)

    pi_image_feats = m.pi_head(image_feats)
    lambda_image_feats = m.lambda_head(image_feats)

    pi_image_feats = F.normalize(pi_image_feats.permute(0, 2, 3, 1), p=2, dim=-1)
    lambda_image_feats = F.normalize(lambda_image_feats.permute(0, 2, 3, 1), p=2, dim=-1)

    pi_logit_map = m.pi_logit_scale.exp() * pi_image_feats @ m.pi_text_feats.t()
    lambda_logit_map = m.lambda_logit_scale.exp() * lambda_image_feats @ m.lambda_text_feats.t()

    pi_logit_map = pi_logit_map.permute(0, 3, 1, 2)          # B, 2, H, W
    lambda_logit_map = lambda_logit_map.permute(0, 3, 1, 2)  # B, N-1, H, W

    pi_prob = pi_logit_map.softmax(dim=1)
    pi_zero_map = pi_prob[:, 0:1]
    pi_nonzero_map = pi_prob[:, 1:2]

    lambda_prob = lambda_logit_map.softmax(dim=1)
    positive_centers = m.bin_centers[:, 1:].to(lambda_prob.device, dtype=lambda_prob.dtype)
    lambda_map = (lambda_prob * positive_centers).sum(dim=1, keepdim=True)

    den_map = pi_nonzero_map * lambda_map

    centers_1d = m.bin_centers.view(-1)[1:].to(lambda_prob.device, dtype=lambda_prob.dtype)
    high_mask = centers_1d >= float(high_bin_min_count)
    if high_mask.sum() == 0:
        high_mask[-1] = True

    lambda_high_mass_map = lambda_prob[:, high_mask].sum(dim=1, keepdim=True)

    # 完整 bin 编号：0 是 zero bin，lambda 分支对应 1..N-1
    full_bin_indices = torch.arange(
        1,
        lambda_prob.shape[1] + 1,
        device=lambda_prob.device,
        dtype=lambda_prob.dtype,
    ).view(1, -1, 1, 1)

    lambda_expected_bin_index_map = (lambda_prob * full_bin_indices).sum(dim=1, keepdim=True)

    return {
        "den_map": den_map,
        "lambda_map": lambda_map,
        "pi_zero_map": pi_zero_map,
        "lambda_high_mass_map": lambda_high_mass_map,
        "lambda_expected_bin_index_map": lambda_expected_bin_index_map,
    }


def get_block_size(model) -> int:
    m = unwrap_model(model)
    return int(m.block_size)


def sliding_window_zip_maps(
    model,
    image: torch.Tensor,
    window_size: int,
    stride: int,
    max_num_windows: int,
    high_bin_min_count: float,
) -> Dict[str, torch.Tensor]:
    assert image.ndim == 4 and image.shape[0] == 1, f"Expected image shape (1,C,H,W), got {tuple(image.shape)}"

    block_size = get_block_size(model)
    image_height, image_width = image.shape[-2:]

    window_height = window_width = int(window_size)
    stride_height = stride_width = int(stride)

    assert image_height >= window_height and image_width >= window_width
    assert window_height % block_size == 0 and window_width % block_size == 0

    num_rows = int(np.ceil((image_height - window_height) / stride_height) + 1)
    num_cols = int(np.ceil((image_width - window_width) / stride_width) + 1)

    windows, coords = [], []

    for i in range(num_rows):
        for j in range(num_cols):
            x_start, y_start = i * stride_height, j * stride_width
            x_end, y_end = x_start + window_height, y_start + window_width

            if x_end > image_height:
                x_start, x_end = image_height - window_height, image_height
            if y_end > image_width:
                y_start, y_end = image_width - window_width, image_width

            windows.append(image[:, :, x_start:x_end, y_start:y_end])
            coords.append((x_start, x_end, y_start, y_end))

    windows = torch.cat(windows, dim=0).to(image.device)

    pred_chunks = []
    for i in range(0, len(windows), max_num_windows):
        maps = zip_forward_maps(model, windows[i:i + max_num_windows], high_bin_min_count)

        pred_chunks.append(torch.cat([
            maps["den_map"],
            maps["lambda_map"],
            maps["pi_zero_map"],
            maps["lambda_high_mass_map"],
            maps["lambda_expected_bin_index_map"],
        ], dim=1).detach().float().cpu())

    preds = torch.cat(pred_chunks, dim=0).numpy()

    out_h, out_w = image_height // block_size, image_width // block_size
    pred_map = np.zeros((5, out_h, out_w), dtype=np.float32)
    count_map = np.zeros((5, out_h, out_w), dtype=np.float32)

    for idx, (x_start, x_end, y_start, y_end) in enumerate(coords):
        xs, xe = x_start // block_size, x_end // block_size
        ys, ye = y_start // block_size, y_end // block_size

        pred_map[:, xs:xe, ys:ye] += preds[idx]
        count_map[:, xs:xe, ys:ye] += 1.0

    pred_map = pred_map / np.maximum(count_map, 1e-6)
    tensor = torch.from_numpy(pred_map).unsqueeze(0).to(image.device)

    return {
        "den_map": tensor[:, 0:1],
        "lambda_map": tensor[:, 1:2],
        "pi_zero_map": tensor[:, 2:3],
        "lambda_high_mass_map": tensor[:, 3:4],
        "lambda_expected_bin_index_map": tensor[:, 4:5],
    }


def maybe_resize_like_test(image: torch.Tensor, points: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对齐 test.py：
    如果图片小于 window_size，则放大图片。
    为了 block 级诊断，点坐标也同步缩放。
    """
    orig_h, orig_w = image.shape[-2:]
    image_h, image_w = orig_h, orig_w
    aspect_ratio = image_w / image_h

    if image_h < window_size:
        new_h = window_size
        new_w = int(new_h * aspect_ratio)
        image = F.interpolate(image, size=(new_h, new_w), mode="bicubic", align_corners=False)
        image_h, image_w = new_h, new_w

    if image_w < window_size:
        new_w = window_size
        new_h = int(new_w / aspect_ratio)
        image = F.interpolate(image, size=(new_h, new_w), mode="bicubic", align_corners=False)
        image_h, image_w = new_h, new_w

    if points.numel() > 0 and (image_h != orig_h or image_w != orig_w):
        points = points.clone()
        points[:, 0] *= image_w / orig_w
        points[:, 1] *= image_h / orig_h

    return image, points


def gt_block_count_map(points: torch.Tensor, map_h: int, map_w: int, block_size: int, device) -> torch.Tensor:
    count_map = torch.zeros((1, 1, map_h, map_w), dtype=torch.float32, device=device)

    if points is None or points.numel() == 0:
        return count_map

    pts = points.to(device=device, dtype=torch.float32)
    xs = torch.div(pts[:, 0], block_size, rounding_mode="floor").long().clamp(0, map_w - 1)
    ys = torch.div(pts[:, 1], block_size, rounding_mode="floor").long().clamp(0, map_h - 1)

    for y, x in zip(ys, xs):
        count_map[0, 0, y, x] += 1.0

    return count_map


def top_quantile_mask(x: torch.Tensor, q: float) -> torch.Tensor:
    flat = x.flatten()
    n = flat.numel()
    k = max(1, int(math.ceil((1.0 - q) * n)))

    idx = torch.topk(flat, k=k, largest=True).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[idx] = True

    return mask.view_as(x)


def nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def mean_on_mask(x: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.sum().item() == 0:
        return float("nan")
    return float(x[mask].mean().detach().cpu())


def bucket_name(gt: float) -> str:
    if gt < 50:
        return "0-50"
    if gt < 150:
        return "50-150"
    if gt < 300:
        return "150-300"
    return "300-inf"


def compute_errors(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    if len(gt) == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "pred_gt_ratio": float("nan"),
        }

    return {
        "mae": float(np.mean(np.abs(pred - gt))),
        "rmse": float(np.sqrt(np.mean((pred - gt) ** 2))),
        "bias": float(np.mean(pred - gt)),
        "pred_gt_ratio": float(np.sum(pred) / max(np.sum(gt), 1e-12)),
    }


def main():
    parser = argparse.ArgumentParser("ZIP bucket and lambda/pi diagnostics")

    parser.add_argument("--dataset", type=str, default="sha")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--sliding_window", action="store_true")
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max_input_size", type=int, default=4096)
    parser.add_argument("--max_num_windows", type=int, default=64)

    parser.add_argument("--high_bin_min_count", type=float, default=10.0)
    parser.add_argument(
        "--high_pred_q",
        type=float,
        default=0.80,
        help="预测密度 top (1-q) 的 block 作为 high_pred_blocks。默认 q=0.80，即 top 20%。",
    )

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    args.dataset = standardize_dataset_name(args.dataset)
    args.stride = args.stride or args.input_size
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.output_dir is None:
        stem = os.path.splitext(os.path.basename(args.weight_path))[0]
        args.output_dir = os.path.join("results", args.dataset, args.split, f"zip_diag_{stem}")

    os.makedirs(args.output_dir, exist_ok=True)

    model = get_model(model_info_path=args.weight_path).to(device)
    model.eval()
    block_size = get_block_size(model)

    dataset = Crowd(
        dataset=args.dataset,
        split=args.split,
        transforms=None,
        sigma=None,
        return_filename=True,
        num_crops=1,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    rows = []

    with torch.no_grad():
        for image, gt_points, _, image_names in tqdm(loader, desc="ZIP diagnostics"):
            filename = image_names[0]

            image = image.to(device)
            points = gt_points[0]

            image, points_scaled = maybe_resize_like_test(image, points, args.input_size)
            h, w = image.shape[-2:]

            with autocast(device_type="cuda", enabled=args.amp and device.type == "cuda"):
                if args.sliding_window or (h * w) > args.max_input_size ** 2:
                    maps = sliding_window_zip_maps(
                        model,
                        image,
                        window_size=args.input_size,
                        stride=args.stride,
                        max_num_windows=args.max_num_windows,
                        high_bin_min_count=args.high_bin_min_count,
                    )
                else:
                    maps = zip_forward_maps(model, image, args.high_bin_min_count)

            den = maps["den_map"].float()
            pi = maps["pi_zero_map"].float()
            high_mass = maps["lambda_high_mass_map"].float()
            eidx = maps["lambda_expected_bin_index_map"].float()

            map_h, map_w = den.shape[-2:]
            gt_block = gt_block_count_map(points_scaled, map_h, map_w, block_size, device=device)

            pos_mask = gt_block > 0
            high_pred_mask = top_quantile_mask(den, args.high_pred_q)

            gt_count = float(len(points))
            pred_count = float(den.sum().detach().cpu())
            ratio = pred_count / gt_count if gt_count > 0 else float("nan")

            row = {
                "filename": filename,
                "bucket": bucket_name(gt_count),
                "gt_count": gt_count,
                "pred_count": pred_count,
                "pred_gt_ratio": ratio,
                "error": pred_count - gt_count,
                "abs_error": abs(pred_count - gt_count),
                "gt_positive_blocks": int(pos_mask.sum().item()),
                "high_pred_blocks": int(high_pred_mask.sum().item()),

                "lambda_high_bin_mass_all": float(high_mass.mean().detach().cpu()),
                "lambda_high_bin_mass_pos": mean_on_mask(high_mass, pos_mask),
                "lambda_high_bin_mass_high_pred": mean_on_mask(high_mass, high_pred_mask),

                "lambda_expected_bin_index_all": float(eidx.mean().detach().cpu()),
                "lambda_expected_bin_index_pos": mean_on_mask(eidx, pos_mask),
                "lambda_expected_bin_index_high_pred": mean_on_mask(eidx, high_pred_mask),

                "pi_mean_all": float(pi.mean().detach().cpu()),
                "pi_mean_on_positive_blocks": mean_on_mask(pi, pos_mask),
                "pi_mean_on_high_pred_blocks": mean_on_mask(pi, high_pred_mask),
            }

            rows.append(row)

    per_image_csv = os.path.join(args.output_dir, "per_image_zip_diag.csv")

    with open(per_image_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    pred_all = np.array([r["pred_count"] for r in rows], dtype=np.float64)
    gt_all = np.array([r["gt_count"] for r in rows], dtype=np.float64)

    summary = {
        "overall": compute_errors(pred_all, gt_all),
        "buckets": {},
    }

    stat_cols = [
        "pred_gt_ratio",
        "lambda_high_bin_mass_all",
        "lambda_high_bin_mass_pos",
        "lambda_high_bin_mass_high_pred",
        "lambda_expected_bin_index_all",
        "lambda_expected_bin_index_pos",
        "lambda_expected_bin_index_high_pred",
        "pi_mean_all",
        "pi_mean_on_positive_blocks",
        "pi_mean_on_high_pred_blocks",
    ]

    for b in ["0-50", "50-150", "150-300", "300-inf"]:
        br = [r for r in rows if r["bucket"] == b]

        pred = np.array([r["pred_count"] for r in br], dtype=np.float64)
        gt = np.array([r["gt_count"] for r in br], dtype=np.float64)

        item = {
            "num_images": len(br),
            **compute_errors(pred, gt),
        }

        item["gt_mean"] = float(np.mean(gt)) if len(gt) else float("nan")
        item["pred_mean"] = float(np.mean(pred)) if len(pred) else float("nan")

        for c in stat_cols:
            item[c] = nanmean([r[c] for r in br])

        summary["buckets"][b] = item

    for c in stat_cols:
        summary["overall"][c] = nanmean([r[c] for r in rows])

    summary["args"] = vars(args)
    summary["block_size"] = block_size

    summary_json = os.path.join(args.output_dir, "summary_zip_diag.json")

    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    bucket_csv = os.path.join(args.output_dir, "bucket_summary_zip_diag.csv")

    fieldnames = [
        "bucket",
        "num_images",
        "gt_mean",
        "pred_mean",
        "mae",
        "rmse",
        "bias",
        "pred_gt_ratio",
        "lambda_high_bin_mass_all",
        "lambda_high_bin_mass_pos",
        "lambda_high_bin_mass_high_pred",
        "lambda_expected_bin_index_all",
        "lambda_expected_bin_index_pos",
        "lambda_expected_bin_index_high_pred",
        "pi_mean_all",
        "pi_mean_on_positive_blocks",
        "pi_mean_on_high_pred_blocks",
    ]

    with open(bucket_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for b, item in summary["buckets"].items():
            row = {"bucket": b}
            for k in fieldnames:
                if k != "bucket":
                    row[k] = item.get(k, float("nan"))
            writer.writerow(row)

    print("\nOverall:")
    print(json.dumps(summary["overall"], indent=2))

    print("\nBucket summary:")
    for b, item in summary["buckets"].items():
        print(b, json.dumps(item, indent=2))

    print(f"\nSaved:\n  {per_image_csv}\n  {bucket_csv}\n  {summary_json}")


if __name__ == "__main__":
    main()