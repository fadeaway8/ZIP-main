import os
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import ToTensor, Normalize
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from turbojpeg import TurboJPEG, TJPF_RGB

import datasets
from datasets.crowd import mean, std
from losses.soft_hdba import gt_hdba_candidate_block_mask, ramp_weight, zip_hdba
from utils import update_loss_info, barrier, reduce_mean


jpeg_decoder = TurboJPEG()


def build_labeled_unlabeled_indices(
    n: int,
    label_percent: float,
    label_seed: int,
) -> Tuple[List[int], List[int]]:
    assert 0 < label_percent <= 100.0

    k = max(1, int(round(n * label_percent / 100.0)))

    g = torch.Generator()
    g.manual_seed(label_seed)

    perm = torch.randperm(n, generator=g).tolist()
    labeled_indices = sorted(perm[:k])
    unlabeled_indices = sorted(perm[k:])

    return labeled_indices, unlabeled_indices


def make_train_transforms(args):
    transforms = [
        datasets.RandomResizedCrop(
            (args.input_size, args.input_size),
            scale=(args.aug_min_scale, args.aug_max_scale),
        ),
        datasets.RandomHorizontalFlip(),
    ]

    if args.aug_brightness > 0 or args.aug_contrast > 0 or args.aug_saturation > 0 or args.aug_hue > 0:
        transforms.append(
            datasets.ColorJitter(
                brightness=args.aug_brightness,
                contrast=args.aug_contrast,
                saturation=args.aug_saturation,
                hue=args.aug_hue,
            )
        )

    if args.aug_blur_prob > 0 and args.aug_kernel_size is not None and args.aug_kernel_size > 0:
        transforms.append(
            datasets.RandomApply(
                [datasets.GaussianBlur(kernel_size=args.aug_kernel_size)],
                p=args.aug_blur_prob,
            )
        )

    if args.aug_saltiness > 0 or args.aug_spiciness > 0:
        transforms.append(
            datasets.PepperSaltNoise(
                saltiness=args.aug_saltiness,
                spiciness=args.aug_spiciness,
            )
        )

    from torchvision.transforms.v2 import Compose
    return Compose(transforms)


class UnlabeledPairCrowd(Dataset):
    """
    无标签数据集。
    每次返回同一张图的 weak / strong 两个视图。

    关键点：
    - weak 和 strong 使用相同几何变换：同一个 crop、同一个 resize、同一个 flip
    - strong 额外使用颜色扰动 / 噪声
    - 不使用 label
    """

    def __init__(self, args, indices: List[int]):
        self.args = args
        self.indices = list(indices)

        self.base = datasets.Crowd(
            dataset=args.dataset,
            split="train",
            transforms=None,
            sigma=None,
            return_filename=False,
            num_crops=1,
        )

        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=mean, std=std)

        self.strong_color = datasets.ColorJitter(
            brightness=args.aug_brightness,
            contrast=args.aug_contrast,
            saturation=args.aug_saturation,
            hue=args.aug_hue,
        )

        self.strong_noise = datasets.PepperSaltNoise(
            saltiness=args.aug_saltiness,
            spiciness=args.aug_spiciness,
        )

    def __len__(self):
        return len(self.indices)

    def _load_image(self, idx: int) -> torch.Tensor:
        image_name = self.base.image_names[idx]
        image_path = os.path.join(self.base.root, "train", "images", image_name)

        with open(image_path, "rb") as f:
            image = jpeg_decoder.decode(f.read(), pixel_format=TJPF_RGB)
            image = self.to_tensor(image)

        return image

    def _same_geometry_transform(self, image: torch.Tensor) -> torch.Tensor:
        out_h = out_w = int(self.args.input_size)
        scale = torch.empty(1).uniform_(self.args.aug_min_scale, self.args.aug_max_scale).item()

        crop_h = int(out_h * scale)
        crop_w = int(out_w * scale)

        in_h, in_w = image.shape[-2:]

        if crop_h > in_h or crop_w > in_w:
            ratio = max(crop_h / in_h, crop_w / in_w)
            resize_h = int(in_h * ratio) + 1
            resize_w = int(in_w * ratio) + 1

            image = TF.resize(
                image,
                [resize_h, resize_w],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            in_h, in_w = image.shape[-2:]

        top = torch.randint(0, in_h - crop_h + 1, (1,)).item()
        left = torch.randint(0, in_w - crop_w + 1, (1,)).item()

        image = TF.crop(image, top, left, crop_h, crop_w)
        image = TF.resize(
            image,
            [out_h, out_w],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

        if torch.rand(1).item() < 0.5:
            image = TF.hflip(image)

        return image

    def _strong_appearance(self, image: torch.Tensor) -> torch.Tensor:
        dummy_label = torch.zeros((0, 2), dtype=torch.float32)

        if self.args.aug_brightness > 0 or self.args.aug_contrast > 0 or self.args.aug_saturation > 0 or self.args.aug_hue > 0:
            image, _ = self.strong_color(image, dummy_label)

        if self.args.aug_saltiness > 0 or self.args.aug_spiciness > 0:
            image, _ = self.strong_noise(image, dummy_label)

        return image

    def __getitem__(self, i: int):
        idx = self.indices[i]
        image = self._load_image(idx)

        image_geom = self._same_geometry_transform(image)

        weak = image_geom.clone()
        strong = self._strong_appearance(image_geom.clone())

        weak = self.normalize(weak)
        strong = self.normalize(strong)

        return weak, strong


def get_ssl_dataloaders(args):
    """
    返回：
    labeled_loader: 10% 有标签数据
    unlabeled_loader: 剩下 90% 无标签数据
    """

    assert args.nprocs == 1, "这份半监督最小版本先只支持单卡。请用 CUDA_VISIBLE_DEVICES=0。"

    train_transforms = make_train_transforms(args)

    dataset_class = datasets.InMemoryCrowd if args.in_memory_dataset else datasets.Crowd

    full_labeled_dataset = dataset_class(
        dataset=args.dataset,
        split="train",
        transforms=train_transforms,
        sigma=None,
        return_filename=False,
        num_crops=args.num_crops,
    )

    n = len(full_labeled_dataset)
    labeled_indices, unlabeled_indices = build_labeled_unlabeled_indices(
        n=n,
        label_percent=args.label_percent,
        label_seed=args.label_seed,
    )

    labeled_dataset = Subset(full_labeled_dataset, labeled_indices)
    unlabeled_dataset = UnlabeledPairCrowd(args, unlabeled_indices)

    prefetch_factor = None if args.num_workers == 0 else 3
    persistent_workers = False if args.num_workers == 0 else True

    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=datasets.collate_fn,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=args.batch_size_u,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    return labeled_loader, unlabeled_loader, labeled_indices, unlabeled_indices


def create_ema_teacher(student: nn.Module) -> nn.Module:
    teacher = deepcopy(student)
    teacher.eval()

    for p in teacher.parameters():
        p.requires_grad_(False)

    return teacher


@torch.no_grad()
def update_ema_teacher(teacher: nn.Module, student: nn.Module, decay: float):
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(decay).add_(s_param.data, alpha=1.0 - decay)

    # BN / running stats 直接同步，更稳
    for t_buf, s_buf in zip(teacher.buffers(), student.buffers()):
        t_buf.copy_(s_buf)


def get_ssl_lambda(args, epoch: int) -> float:
    if epoch <= args.ssl_warmup_epochs:
        return 0.0

    if args.ssl_rampup_epochs <= 0:
        return args.lambda_u_max

    progress = (epoch - args.ssl_warmup_epochs) / float(args.ssl_rampup_epochs)
    progress = max(0.0, min(1.0, progress))

    # sigmoid ramp-up
    ramp = float(np.exp(-5.0 * (1.0 - progress) ** 2))

    return args.lambda_u_min + (args.lambda_u_max - args.lambda_u_min) * ramp


def forward_den_map(model: nn.Module, image: torch.Tensor):
    """
    返回最终预测密度图 pred_den_map。

    兼容两种情况：
    1. train 模式：ZIP 返回 4 个值，非 ZIP 返回 2 个值
    2. eval 模式：模型直接返回 den_map
    """
    out = model(image)

    # eval 模式下，CLIP_EBC / ZIP 直接返回 den_map
    if torch.is_tensor(out):
        return out

    # train 模式下，返回 tuple/list，最后一个通常是 den_map
    if isinstance(out, (tuple, list)):
        return out[-1]

    raise TypeError(f"Unexpected model output type: {type(out)}")


def density_consistency_loss(
    student_den: torch.Tensor,
    teacher_den: torch.Tensor,
):
    """
    几何已经对齐，所以可以直接做 map consistency。

    map loss:
      sum(|student - teacher|) / (sum(teacher) + 1)

    count loss:
      |sum(student) - sum(teacher)| / (sum(teacher) + 1)
    """
    teacher_den = teacher_den.detach().float()
    student_den = student_den.float()

    s_flat = student_den.flatten(1)
    t_flat = teacher_den.flatten(1)

    t_count = t_flat.sum(dim=1).detach()
    s_count = s_flat.sum(dim=1)

    map_loss = (torch.abs(s_flat - t_flat).sum(dim=1) / (t_count + 1.0)).mean()
    count_loss = (torch.abs(s_count - t_count) / (t_count + 1.0)).mean()

    return map_loss, count_loss


def cycle_next(iterator, loader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def train_semi(
    model: nn.Module,
    teacher_model: nn.Module,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    grad_scaler: GradScaler,
    device: torch.device,
    epoch: int,
    args,
):
    model.train()
    teacher_model.eval()

    info = None

    n_steps = max(len(labeled_loader), len(unlabeled_loader))
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    lambda_u = get_ssl_lambda(args, epoch)
    hdba_w = ramp_weight(
        epoch=epoch,
        max_weight=getattr(args, "hdba_w", 0.0),
        warmup_epochs=getattr(args, "hdba_warmup_epochs", 0),
        ramp_epochs=getattr(args, "hdba_ramp_epochs", 0),
    )

    data_iter = tqdm(range(n_steps)) if args.local_rank == 0 else range(n_steps)

    for _ in data_iter:
        labeled_batch, labeled_iter = cycle_next(labeled_iter, labeled_loader)
        unlabeled_batch, unlabeled_iter = cycle_next(unlabeled_iter, unlabeled_loader)

        image_l, gt_points_l, gt_den_map_l = labeled_batch
        image_u_w, image_u_s = unlabeled_batch

        image_l = image_l.to(device)
        gt_points_l = [p.to(device) for p in gt_points_l]
        gt_den_map_l = gt_den_map_l.to(device)

        image_u_w = image_u_w.to(device)
        image_u_s = image_u_s.to(device)

        with torch.set_grad_enabled(True):
            with autocast(device_type="cuda", enabled=grad_scaler is not None and grad_scaler.is_enabled()):
                # supervised branch
                if model.zero_inflated:
                    pred_logit_pi_map, pred_logit_map, pred_lambda_map, pred_den_map = model(image_l)

                    sup_loss, sup_info = loss_fn(
                        pred_logit_pi_map=pred_logit_pi_map,
                        pred_logit_map=pred_logit_map,
                        pred_lambda_map=pred_lambda_map,
                        pred_den_map=pred_den_map,
                        gt_den_map=gt_den_map_l,
                        gt_points=gt_points_l,
                    )
                    block_mask = gt_hdba_candidate_block_mask(
                        gt_den_map=gt_den_map_l,
                        gt_points=gt_points_l,
                        input_size=loss_fn.input_size,
                        block_size=model.block_size,
                        out_h=pred_den_map.shape[-2],
                        out_w=pred_den_map.shape[-1],
                        img_count_thr=args.hdba_img_count_thr,
                        top_ratio=args.hdba_top_ratio,
                        block_count_thr=args.hdba_block_count_thr,
                    )
                    hdba_loss, hdba_info = zip_hdba(
                        model=model,
                        lambda_logit=pred_logit_map,
                        pi_logit=pred_logit_pi_map,
                        block_mask=block_mask,
                        high_bin_min_count=args.high_bin_min_count,
                        margin=args.hdba_margin,
                        pi_nonzero_thr=args.pi_nonzero_thr,
                    )
                else:
                    pred_logit_map, pred_den_map = model(image_l)

                    sup_loss, sup_info = loss_fn(
                        pred_logit_map=pred_logit_map,
                        pred_den_map=pred_den_map,
                        gt_den_map=gt_den_map_l,
                        gt_points=gt_points_l,
                    )
                    hdba_loss = pred_den_map.mean() * 0.0
                    hdba_info = {}

                # unlabeled teacher/student branch
                with torch.no_grad():
                    teacher_den = forward_den_map(teacher_model, image_u_w)

                u_hdba_loss = hdba_loss * 0.0
                if model.zero_inflated:
                    pred_logit_pi_map_u, pred_logit_map_u, _, student_den = model(image_u_s)
                    if getattr(args, "apply_on_unlabeled", False) and hdba_w > 0:
                        with torch.no_grad():
                            t_quantile = torch.quantile(
                                teacher_den.float().flatten(1),
                                args.hdba_block_q,
                                dim=1,
                            ).view(-1, 1, 1, 1)
                            u_mask = teacher_den >= t_quantile
                        u_hdba_loss, _ = zip_hdba(
                            model=model,
                            lambda_logit=pred_logit_map_u,
                            pi_logit=pred_logit_pi_map_u,
                            block_mask=u_mask,
                            high_bin_min_count=args.high_bin_min_count,
                            margin=args.hdba_margin,
                            pi_nonzero_thr=args.pi_nonzero_thr,
                        )
                else:
                    student_den = forward_den_map(model, image_u_s)

                ssl_map_loss, ssl_count_loss = density_consistency_loss(
                    student_den=student_den,
                    teacher_den=teacher_den,
                )

                ssl_loss = args.ssl_map_w * ssl_map_loss + args.ssl_count_w * ssl_count_loss
                total_loss = (
                    sup_loss
                    + lambda_u * ssl_loss
                    + hdba_w * (hdba_loss + args.hdba_u_w * u_hdba_loss)
                )

        optimizer.zero_grad()

        if grad_scaler is not None:
            grad_scaler.scale(total_loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        update_ema_teacher(
            teacher=teacher_model,
            student=model,
            decay=args.ema_decay,
        )

        loss_info = dict(sup_info)
        loss_info["ssl_map_loss"] = ssl_map_loss.detach()
        loss_info["ssl_count_loss"] = ssl_count_loss.detach()
        loss_info["ssl_loss"] = ssl_loss.detach()
        loss_info["lambda_u"] = torch.tensor(lambda_u, device=device)
        loss_info["hdba_loss"] = hdba_loss.detach()
        loss_info["hdba_u_loss"] = u_hdba_loss.detach()
        loss_info["hdba_w"] = torch.tensor(hdba_w, device=device)
        loss_info.update(hdba_info)
        loss_info["total_loss"] = total_loss.detach()

        loss_info = {
            k: reduce_mean(v.detach(), args.nprocs).item() if args.nprocs > 1 else v.detach().item()
            for k, v in loss_info.items()
        }

        info = update_loss_info(info, loss_info)
        barrier(args.nprocs > 1)

    torch.cuda.empty_cache()

    return model, teacher_model, optimizer, grad_scaler, {k: np.mean(v) for k, v in info.items()}
