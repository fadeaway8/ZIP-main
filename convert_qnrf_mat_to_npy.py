from pathlib import Path
import argparse
import shutil
import re
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm


def norm_stem(path: Path) -> str:
    """
    将 img_0001_ann.mat、GT_img_0001.mat、img_0001.jpg
    统一成 img_0001，便于图像和标注匹配。
    """
    stem = path.stem

    if stem.startswith("GT_"):
        stem = stem[3:]

    for suffix in ["_ann", "-ann", "_gt", "-gt"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    return stem


def get_numeric_id(path: Path):
    nums = re.findall(r"\d+", path.stem)
    if not nums:
        return path.stem
    return int(nums[-1])


def load_qnrf_points(mat_path: Path) -> np.ndarray:
    """
    UCF-QNRF 原始 .mat 通常包含 annPoints，形状为 N x 2。
    输出为 float32 的 N x 2 点坐标数组。
    """
    mat = loadmat(str(mat_path))

    if "annPoints" in mat:
        points = mat["annPoints"]
    else:
        # 兜底：自动寻找形状像 N x 2 或 2 x N 的数值数组
        candidates = []
        for k, v in mat.items():
            if k.startswith("__"):
                continue
            arr = np.asarray(v)
            if arr.ndim == 2 and 2 in arr.shape and arr.size > 0:
                candidates.append((k, arr))

        if not candidates:
            raise KeyError(
                f"{mat_path} 中没有找到 annPoints，也没有找到形状类似 N x 2 的点坐标数组。"
            )

        # 优先选列数为 2 的数组
        candidates.sort(key=lambda x: 0 if x[1].shape[1] == 2 else 1)
        key, points = candidates[0]
        print(f"[提示] {mat_path.name} 没有 annPoints，使用字段 {key}。")

    points = np.asarray(points)

    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    if points.ndim != 2:
        raise ValueError(f"{mat_path} 的点坐标维度异常: {points.shape}")

    if points.shape[1] == 2:
        pass
    elif points.shape[0] == 2:
        points = points.T
    else:
        raise ValueError(f"{mat_path} 的点坐标形状不是 N x 2: {points.shape}")

    return points.astype(np.float32)


def convert_split(src_split_dir: Path, dst_split_dir: Path, expected_num: int):
    image_out = dst_split_dir / "images"
    label_out = dst_split_dir / "labels"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    images = sorted(src_split_dir.glob("*.jpg"), key=get_numeric_id)
    mats = sorted(src_split_dir.glob("*.mat"), key=get_numeric_id)

    print(f"\n处理 {src_split_dir}")
    print(f"发现图片: {len(images)}")
    print(f"发现 .mat 标注: {len(mats)}")

    if len(images) != expected_num:
        raise RuntimeError(f"{src_split_dir} 图片数量应为 {expected_num}，实际为 {len(images)}")

    if len(mats) != expected_num:
        raise RuntimeError(f"{src_split_dir} 标注数量应为 {expected_num}，实际为 {len(mats)}")

    mat_map = {norm_stem(p): p for p in mats}

    missing = []
    for img in images:
        key = norm_stem(img)
        if key not in mat_map:
            missing.append(img.name)

    if missing:
        print("以下图片没有匹配到 .mat 标注：")
        for name in missing[:20]:
            print(name)
        raise RuntimeError(f"共有 {len(missing)} 张图片没有匹配到标注。")

    for img in tqdm(images, desc=f"转换 {src_split_dir.name}"):
        key = norm_stem(img)
        mat_path = mat_map[key]

        # 复制图片，文件名保持不变
        dst_img = image_out / img.name
        if not dst_img.exists():
            shutil.copy2(img, dst_img)

        # 标签名必须和图片编号对应。推荐直接用图片 stem。
        points = load_qnrf_points(mat_path)
        np.save(label_out / f"{img.stem}.npy", points)

    out_images = list(image_out.glob("*.jpg"))
    out_labels = list(label_out.glob("*.npy"))

    print(f"输出图片: {len(out_images)}")
    print(f"输出标签: {len(out_labels)}")

    if len(out_images) != expected_num or len(out_labels) != expected_num:
        raise RuntimeError("输出数量不正确，请检查。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_root",
        type=str,
        required=True,
        help="原始 UCF-QNRF 根目录，里面应包含 Train 和 Test 文件夹。",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="data/qnrf",
        help="输出到 ZIP 代码需要的数据目录，默认 data/qnrf。",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)

    train_dir = raw_root / "Train"
    test_dir = raw_root / "Test"

    if not train_dir.exists():
        raise FileNotFoundError(f"找不到训练目录: {train_dir}")

    if not test_dir.exists():
        raise FileNotFoundError(f"找不到测试目录: {test_dir}")

    convert_split(train_dir, out_root / "train", expected_num=1201)
    convert_split(test_dir, out_root / "val", expected_num=334)

    print("\nUCF-QNRF 转换完成。")
    print(f"输出目录: {out_root.resolve()}")


if __name__ == "__main__":
    main()