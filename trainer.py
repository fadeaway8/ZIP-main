import torch
from torch import nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler
import numpy as np
from copy import deepcopy
from argparse import ArgumentParser
import os, json, hashlib, yaml

current_dir = os.path.abspath(os.path.dirname(__file__))

from datasets import standardize_dataset_name
from models import get_model

from utils import setup, cleanup, init_seeds, get_logger, get_config, barrier
from utils import get_dataloader, get_loss_fn, get_optimizer, load_checkpoint, save_checkpoint
from utils import get_writer, update_train_result, update_eval_result, log, calc_bin_center
from utils.semi_utils import get_ssl_dataloaders, create_ema_teacher, train_semi
from train import train
from evaluate import evaluate


parser = ArgumentParser(description="Train an EBC model.")

# Parameters for model
parser.add_argument("--model_name", type=str, default="CLIP_RN50", help="The model to train.")
parser.add_argument("--block_size", type=int, default=16, choices=[7, 8, 14, 16, 28, 32], help="The block sizes for the model.")
parser.add_argument("--clip_weight_name", type=str, default=None, help="The weight name for CLIP models.")
parser.add_argument("--norm", type=str, default="none", choices=["none", "bn", "ln"], help="The normalization layer to use. 'none' means no normalization layer will be detected automatically, 'bn' means batch normalization, 'ln' means layer normalization.")
parser.add_argument("--act", type=str, default="none", choices=["none", "relu", "gelu"], help="The activation function to use. 'none' means no activation function will be detected automatically, 'relu' means ReLU, 'gelu' means GELU.")

parser.add_argument("--num_vpt", type=int, default=96, help="The number of visual prompt tokens.")
parser.add_argument("--vpt_drop", type=float, default=0.0, help="The dropout rate for visual prompt tokens.")

parser.add_argument("--adapter", action="store_true", help="Use adapter for the model. This will freeze the backbone and only train the adapter layers and newly added layers.")
parser.add_argument("--adapter_reduction", type=int, default=4, help="The reduction ratio for the adapter layers. This will be used to reduce the number of parameters in the adapter layers.")

parser.add_argument("--lora", action="store_true", help="Use LoRA for the model. This will freeze the backbone and only train the LoRA layers and newly added layers.")
parser.add_argument("--lora_rank", type=int, default=16, help="The rank for the LoRA layers. This will be used to reduce the number of parameters in the LoRA layers.")
parser.add_argument("--lora_alpha", type=float, default=32.0, help="The alpha for the LoRA layers. This will be used to scale the LoRA layers.")
parser.add_argument("--lora_dropout", type=float, default=0.0, help="The dropout rate for the LoRA layers.")

# Parameters for dataset
parser.add_argument("--dataset", type=str, required=True, help="The dataset to train on.")
parser.add_argument("--in_memory_dataset", action="store_true", help="Load the dataset into memory. This will speed up training but requires more memory.")
parser.add_argument("--input_size", type=int, default=None, help="The size of the input image.")
parser.add_argument("--batch_size", type=int, default=None, help="The training batch size.")
parser.add_argument("--num_crops", type=int, default=None, help="The number of crops for multi-crop training.")
parser.add_argument("--aug_min_scale", type=float, default=None, help="The minimum scale for random scale augmentation.")
parser.add_argument("--aug_max_scale", type=float, default=None, help="The maximum scale for random scale augmentation.")
parser.add_argument("--aug_brightness", type=float, default=None, help="The brightness factor for random color jitter augmentation.")
parser.add_argument("--aug_contrast", type=float, default=None, help="The contrast factor for random color jitter augmentation.")
parser.add_argument("--aug_saturation", type=float, default=None, help="The saturation factor for random color jitter augmentation.")
parser.add_argument("--aug_hue", type=float, default=None, help="The hue factor for random color jitter augmentation.")
parser.add_argument("--aug_kernel_size", type=int, default=None, help="The kernel size for Gaussian blur augmentation.")
parser.add_argument("--aug_saltiness", type=float, default=None, help="The saltiness for pepper salt noise augmentation.")
parser.add_argument("--aug_spiciness", type=float, default=None, help="The spiciness for pepper salt noise augmentation.")
parser.add_argument("--aug_blur_prob", type=float, default=None, help="The probability for Gaussian blur augmentation.")

# Parameters for evaluation
parser.add_argument("--sliding_window", action="store_true", help="Use sliding window strategy for evaluation.")
parser.add_argument("--stride", type=int, default=None, help="The stride for sliding window strategy.")
parser.add_argument("--max_input_size", type=int, default=4096, help="The maximum size of the input image in evaluation. Images larger than this will be processed using sliding window by force to avoid OOM.")
parser.add_argument("--max_num_windows", type=int, default=64, help="The maximum number of windows to be simultaneously processed.")
parser.add_argument("--resize_to_multiple", action="store_true", help="Resize the image to a multiple of the input size.")

# Parameters for loss function
parser.add_argument("--reg_loss", type=str, default="zipnll", choices=["zipnll", "pnll", "dm", "msmae", "mae", "mse"], help="The regression loss function.")
parser.add_argument("--aux_loss", type=str, default="msmae", choices=["zipnll", "pnll", "dm", "msmae", "mae", "mse", "none"], help="The auxiliary loss function.")
parser.add_argument("--weight_cls", type=float, default=1.0, help="The weight for classification loss.")
parser.add_argument("--weight_reg", type=float, default=1.0, help="The weight for regression loss.")
parser.add_argument("--weight_aux", type=float, default=1.0, help="The weight for auxiliary loss.")
parser.add_argument("--numItermax", type=int, default=100, help="The maximum number of iterations for the OT/POT solver.")
parser.add_argument("--regularization", type=float, default=10.0, help="The regularization term for the OT/POT loss.")
parser.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4], help="The scales for multi-scale mae loss.")
parser.add_argument("--min_scale_weight", type=float, default=0.0, help="The minimum weight for multi-scale mae loss.")
parser.add_argument("--max_scale_weight", type=float, default=1.0, help="The maximum weight for multi-scale mae loss.")
parser.add_argument("--alpha", type=float, default=0.5, help="The alpha for multi-scale mae loss.")

# Parameters for Soft HDBA on ZIP lambda branch
parser.add_argument("--hdba_w", type=float, default=0.005, help="Maximum weight for Soft HDBA loss. Set to 0 to disable.")
parser.add_argument("--high_bin_min_count", type=float, default=10.0, help="Minimum positive bin center treated as a high-count bin for HDBA.")
parser.add_argument("--hdba_block_q", type=float, default=0.8, help="Teacher density quantile for unlabeled HDBA block selection.")
parser.add_argument("--hdba_margin", type=float, default=0.02, help="Margin for high-bin probability mass in HDBA.")
parser.add_argument("--hdba_warmup_epochs", type=int, default=100, help="Epochs before enabling HDBA.")
parser.add_argument("--hdba_ramp_epochs", type=int, default=0, help="Epochs to linearly ramp HDBA weight after warmup.")
parser.add_argument("--hdba_img_count_thr", type=float, default=300.0, help="Use HDBA only for labeled images with at least this many GT points.")
parser.add_argument("--hdba_top_ratio", type=float, default=0.2, help="Select this top fraction of positive GT blocks per high-count image for supervised HDBA.")
parser.add_argument("--hdba_block_count_thr", type=float, default=0.0, help="If positive, select supervised HDBA blocks whose GT block count is at least this threshold instead of top-ratio selection.")
parser.add_argument("--hdba_u_w", type=float, default=1.0, help="Relative weight for unlabeled HDBA when apply_on_unlabeled is enabled.")
parser.add_argument("--apply_on_unlabeled", action="store_true", help="Apply HDBA to teacher-selected unlabeled blocks in semi-supervised training.")
parser.add_argument("--pi_nonzero_thr", type=float, default=0.4, help="Minimum non-zero probability for the optional HDBA pi gate.")

# Parameters for optimizer
parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam", "adamw", "radam"], help="The optimizer to use.")
parser.add_argument("--lr", type=float, default=None, help="The learning rate for untrained modules.")
parser.add_argument("--vpt_lr", type=float, default=None, help="The learning rate for the visual prompt tokens.")
parser.add_argument("--adapter_lr", type=float, default=None, help="The learning rate for the adapter layers. If None, it will be set to the same as lr.")
parser.add_argument("--lora_lr", type=float, default=None, help="The learning rate for the LoRA layers. If None, it will be set to the same as lr.")
parser.add_argument("--backbone_lr", type=float, default=None, help="The learning rate for the pretrained backbone.")
parser.add_argument("--weight_decay", type=float, default=None, help="The weight decay for untrained modules.")
parser.add_argument("--vpt_weight_decay", type=float, default=None, help="The weight decay for the visual prompt tokens.")
parser.add_argument("--adapter_weight_decay", type=float, default=None, help="The weight decay for the adapter layers. If None, it will be set to the same as weight_decay.")
parser.add_argument("--lora_weight_decay", type=float, default=None, help="The weight decay for the LoRA layers. If None, it will be set to the same as weight_decay.")
parser.add_argument("--backbone_weight_decay", type=float, default=None, help="The weight decay for the pretrained backbone.")

# Parameters for learning rate scheduler
parser.add_argument("--scheduler", type=str, default="cos_restarts", choices=["step", "cos", "cos_restarts"], help="The learning rate scheduler.")
parser.add_argument("--warmup_epochs", type=int, default=25, help="Number of epochs for warmup. The learning rate will linearly change from warmup_lr to lr.")
parser.add_argument("--warmup_lr", type=float, default=1e-5, help="Learning rate for warmup.")
parser.add_argument("--eta_min", type=float, default=1e-6, help="Minimum learning rate.")
# Step Decay parameters
parser.add_argument("--gamma", type=float, default=0.925, help="The decay factor for step scheduler.")
parser.add_argument("--step_size", type=int, default=20, help="The step size for step scheduler.")
# Cosine Annealing with Warm Restarts parameters
parser.add_argument("--T_0", type=int, default=5, help="Number of epochs for the first restart.")
parser.add_argument("--T_mult", type=int, default=2, help="A factor increases T_0 after a restart.")
# Cosine Annealing parameters
parser.add_argument("--T_max", type=int, default=20, help="The maximum number of epochs for the cosine annealing scheduler.")

# Parameters for training
parser.add_argument("--ckpt_dir_name", type=str, default=None, help="The name of the checkpoint folder.")
parser.add_argument("--total_epochs", type=int, default=1300, help="Number of epochs to train.")
parser.add_argument("--eval_start", type=int, default=None, help="Start to evaluate after this number of epochs.")
parser.add_argument("--eval_freq", type=float, default=None, help="Evaluate every this number of epochs. If < 1, evaluate every this fraction of an epoch.")
parser.add_argument("--save_freq", type=int, default=50, help="Save checkpoint every this number of epochs. Could help reduce I/O.")
parser.add_argument("--save_best_k", type=int, default=5, help="Save the best k checkpoints.")
parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision training.")
parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of workers for data loading.")
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")
parser.add_argument("--label_percent", type=float, default=100.0, help="Percentage of labeled training data to use.")
parser.add_argument("--label_seed", type=int, default=42, help="Random seed for labeled subset sampling.")

# Parameters for semi-supervised training
parser.add_argument("--semi", action="store_true", help="Use semi-supervised training with unlabeled data.")
parser.add_argument("--batch_size_u", type=int, default=8, help="Batch size for unlabeled images.")

parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay for teacher model.")

parser.add_argument("--lambda_u_max", type=float, default=0.1, help="Maximum weight for unlabeled consistency loss.")
parser.add_argument("--lambda_u_min", type=float, default=0.0, help="Minimum weight for unlabeled consistency loss.")
parser.add_argument("--ssl_warmup_epochs", type=int, default=100, help="Epochs before enabling SSL loss.")
parser.add_argument("--ssl_rampup_epochs", type=int, default=200, help="Ramp-up epochs for SSL loss.")

parser.add_argument("--ssl_map_w", type=float, default=1.0, help="Weight for unlabeled density-map consistency.")
parser.add_argument("--ssl_count_w", type=float, default=0.5, help="Weight for unlabeled count consistency.")

def run(local_rank: int, nprocs: int, args: ArgumentParser) -> None:
    print(f"Rank {local_rank} process among {nprocs} processes.")
    init_seeds(args.seed + local_rank)
    setup(local_rank, nprocs)
    args.local_rank = local_rank
    print(f"Initialized successfully. Training with {nprocs} GPUs.")
    device = f"cuda:{local_rank}" if local_rank != -1 else "cuda:0"
    print(f"Using device: {device}.")

    ddp = nprocs > 1

    # Define the bins and bin centers
    with open(os.path.join(current_dir, "configs", "bin_config.json"), "r") as f:
        bins = json.load(f)[args.dataset][str(args.block_size)]
    bins = [(float(b[0]), float(b[1])) for b in bins]

    with open(os.path.join(current_dir, "counts", f"{args.dataset}.json"), "r") as f:
        count_stats = json.load(f)[str(args.block_size)]
        count_stats = {int(k): int(v) for k, v in count_stats.items()}
        bin_centers, bin_counts = calc_bin_center(bins, count_stats)

    args.bins = bins
    args.bin_centers = bin_centers
    args.bin_counts = bin_counts

    model = get_model(
        model_info_path=os.path.join(args.ckpt_dir, "model_info.pth"),
        model_name=args.model_name,
        block_size=args.block_size,
        bins=bins,
        bin_centers=bin_centers,
        zero_inflated=args.reg_loss == "zipnll" or args.aux_loss == "zipnll",
        clip_weight_name=args.clip_weight_name,
        num_vpt=args.num_vpt,
        vpt_drop=args.vpt_drop,
        adapter=args.adapter,
        adapter_reduction=args.adapter_reduction,
        lora=args.lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        input_size=args.input_size,
        norm=args.norm,
        act=args.act,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_nontrainable_params = total_params - total_trainable_params

    grad_scaler = GradScaler(device=device) if args.amp else None

    loss_fn = get_loss_fn(args)
    optimizer, scheduler = get_optimizer(args, model)

    model, optimizer, scheduler, grad_scaler, start_epoch, loss_info, hist_val_scores, best_val_scores = load_checkpoint(args, model, optimizer, scheduler, grad_scaler)
    model = DDP(nn.SyncBatchNorm.convert_sync_batchnorm(model), device_ids=[local_rank], output_device=local_rank) if ddp else model

    if local_rank == 0:
        writer = get_writer(args.ckpt_dir)
        logger = get_logger(os.path.join(args.ckpt_dir, "train.log"))
        logger.info(get_config(vars(args), mute=False))
        logger.info(f"Total parameters: {total_params:,}\nTrainable parameters: {total_trainable_params:,}\nNon-trainable parameters: {total_nontrainable_params:,}\n")

    if args.semi:
        train_loader, unlabeled_loader, labeled_indices, unlabeled_indices = get_ssl_dataloaders(args)
        sampler = None

        teacher_model = create_ema_teacher(model)

        if local_rank == 0:
            logger.info(
                f"Semi-supervised training enabled. "
                f"Labeled images: {len(labeled_indices)}, "
                f"Unlabeled images: {len(unlabeled_indices)}, "
                f"label_percent={args.label_percent}, "
                f"label_seed={args.label_seed}"
            )
    else:
        train_loader, sampler = get_dataloader(args, split="train")
        unlabeled_loader = None
        teacher_model = None

    val_loader = get_dataloader(args, split="val")

    for epoch in range(start_epoch, args.total_epochs + 1):  # start from 1
        if local_rank == 0:
            message = f"\tlr: {optimizer.param_groups[0]['lr']:.3e}"
            log(logger, epoch, args.total_epochs, message=message)

        if sampler is not None:
            sampler.set_epoch(epoch)

        if args.semi:
            model, teacher_model, optimizer, grad_scaler, loss_info = train_semi(
                model=model,
                teacher_model=teacher_model,
                labeled_loader=train_loader,
                unlabeled_loader=unlabeled_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                grad_scaler=grad_scaler,
                device=device,
                epoch=epoch,
                args=args,
            )

            scheduler.step()
            barrier(ddp)

            eval_model = (epoch >= args.eval_start) and ((epoch - args.eval_start) % args.eval_freq == 0)

            if eval_model:
                curr_val_scores = evaluate(
                    model=model,
                    data_loader=val_loader,
                    sliding_window=args.sliding_window,
                    max_input_size=args.max_input_size,
                    window_size=args.input_size,
                    stride=args.stride,
                    max_num_windows=args.max_num_windows,
                    device=device,
                    amp=args.amp,
                    local_rank=local_rank,
                    nprocs=nprocs,
                )

                state_dict = deepcopy(model.module.state_dict() if ddp else model.state_dict())
                curr_weights = {k: state_dict for k in curr_val_scores.keys()}
        else:
            if args.eval_freq < 1:
                eval_model = epoch >= args.eval_start

                if eval_model:
                    model, optimizer, grad_scaler, loss_info, curr_val_scores, curr_weights = train(
                        model=model,
                        data_loader=train_loader,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        grad_scaler=grad_scaler,
                        device=device,
                        rank=local_rank,
                        nprocs=nprocs,
                        eval_data_loader=val_loader,
                        eval_freq=args.eval_freq,
                        sliding_window=args.sliding_window,
                        max_input_size=args.max_input_size,
                        window_size=args.input_size,
                        stride=args.stride,
                        max_num_windows=args.max_num_windows,
                        epoch=epoch,
                        args=args,
                    )
                    scheduler.step()
                    barrier(ddp)
                
                else:
                    model, optimizer, grad_scaler, loss_info, _, _ = train(
                        model=model,
                        data_loader=train_loader,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        grad_scaler=grad_scaler,
                        device=device,
                        rank=local_rank,
                        nprocs=nprocs,
                        epoch=epoch,
                        args=args,
                    )
                    scheduler.step()
                    barrier(ddp)

            else:
                model, optimizer, grad_scaler, loss_info, _, _ = train(
                    model=model,
                    data_loader=train_loader,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    grad_scaler=grad_scaler,
                    device=device,
                    rank=local_rank,
                    nprocs=nprocs,
                    epoch=epoch,
                    args=args,
                )

                scheduler.step()
                barrier(ddp)

                eval_model = (epoch >= args.eval_start) and ((epoch - args.eval_start) % args.eval_freq == 0)
                if eval_model:
                    curr_val_scores = evaluate(
                        model=model,
                        data_loader=val_loader,
                        sliding_window=args.sliding_window,
                        max_input_size=args.max_input_size,
                        window_size=args.input_size,
                        stride=args.stride,
                        max_num_windows=args.max_num_windows,
                        device=device,
                        amp=args.amp,
                        local_rank=local_rank,
                        nprocs=nprocs
                    )

                    state_dict = deepcopy(model.module.state_dict() if ddp else model.state_dict())
                    curr_weights = {k: state_dict for k in curr_val_scores.keys()}  # copy the state_dict                    

        if local_rank == 0:
            update_train_result(epoch, loss_info, writer)
            log(logger, None, None, loss_info=loss_info, message="\n" * 2 if not eval_model else None)

            if eval_model:
                hist_val_scores, best_val_scores = update_eval_result(
                    epoch=epoch,
                    curr_scores=curr_val_scores,
                    hist_scores=hist_val_scores,
                    best_scores=best_val_scores,
                    model_info={"config": model.module.config if ddp else model.config, "weights": curr_weights},
                    writer=writer,
                    ckpt_dir=args.ckpt_dir,
                )
        
                log(logger, None, None, None, curr_val_scores, best_val_scores, message="\n" * 3)

        if local_rank == 0 and (epoch % args.save_freq == 0):
            save_checkpoint(
                epoch + 1,
                model.module.state_dict() if ddp else model.state_dict(),
                optimizer.state_dict(),
                scheduler.state_dict() if scheduler is not None else None,
                grad_scaler.state_dict() if grad_scaler is not None else None,
                loss_info,
                hist_val_scores,
                best_val_scores,
                args.ckpt_dir,
            )

        barrier(ddp)

    if local_rank == 0:
        writer.close()
        print("Training completed. Best scores:")
        for k in best_val_scores.keys():
            scores = " ".join([f"{best_val_scores[k][i]:.4f};" for i in range(len(best_val_scores[k]))])
            print(f"    {k}: {scores}. \t Mean: {np.mean(best_val_scores[k]):.4f}")

    cleanup(ddp)


def main():
    args = parser.parse_args()
    args.dataset = standardize_dataset_name(args.dataset)

    dataset_config_path = os.path.join(current_dir, "configs", f"{args.dataset}.yaml")
    with open(dataset_config_path, "r") as f:
        dataset_config = yaml.safe_load(f)
    for k, v in dataset_config.items():
        if k in vars(args) and vars(args)[k] is None:
            vars(args)[k] = v
    
    # Sliding window prediction will be used if args.sliding_window is True, or when the image size is larger than args.max_input_size
    args.stride = args.stride or args.input_size

    assert args.model_name in ["ebc_p", "ebc_n", "ebc_t", "ebc_s", "ebc_b"], f"Expected model_name to be one of ['ebc_p', 'ebc_n', 'ebc_t', 'ebc_s', 'ebc_b'], got {args.model_name}."

    if args.model_name == "ebc_p":  # pico
        args.model_name = "mobilenetv4_conv_small_050"

    elif args.model_name == "ebc_n":  # nano
        args.model_name = "mobilenetv4_conv_small"

    elif args.model_name == "ebc_t": # tiny
        args.model_name = "mobilenetv4_conv_medium"

    elif args.model_name == "ebc_s":
        args.model_name = "CLIP_MobileCLIP_S1"
        args.clip_weight_name = "datacompdr"



    else:  # args.model_name == "ebc_b":

        if args.dataset == "sha":

            args.model_name = "CLIP_ViT_B_16"

            args.clip_weight_name = "openai"

            args.num_vpt = args.num_vpt or 96

        elif args.dataset == "shb":

            args.model_name = "CLIP_ViT_B_16"

            args.clip_weight_name = "openai"
            args.num_vpt = args.num_vpt or 96

        elif args.dataset == "qnrf":

            args.model_name = "CLIP_ViT_B_16"

            args.clip_weight_name = "openai"

            args.num_vpt = args.num_vpt or 96

        else:

            args.model_name = "CLIP_convnext_base_w_320"

            args.clip_weight_name = "laion_aesthetic_s13b_b82k_augreg"

    if "CLIP_" not in args.model_name:
        args.clip_weight_name = None

    if args.adapter:
        assert not args.lora, "Cannot use both adapter and LoRA at the same time."

        args.num_vpt = None
        args.vpt_drop = None
        args.vpt_lr = None
        args.vpt_weight_decay = None
        args.lora_rank = None
        args.lora_alpha = None
        args.lora_dropout = None
        args.lora_lr = None
        args.lora_weight_decay = None
        args.backbone_lr = None
        args.backbone_weight_decay = None

        assert args.adapter_lr > 0, f"Expected adapter_lr to be greater than 0, got {args.adapter_lr}"
        assert args.adapter_weight_decay > 0, f"Expected adapter_weight_decay to be greater than 0, got {args.adapter_weight_decay}"
        assert args.adapter_reduction > 0, f"Expected adapter_reduction to be greater than 0, got {args.adapter_reduction}"
    
    else:
        args.adapter_reduction = None
        args.adapter_lr = None
        args.adapter_weight_decay = None
    
    if args.lora:
        assert not args.adapter, "Cannot use both adapter and LoRA at the same time."

        args.num_vpt = None
        args.vpt_drop = None
        args.vpt_lr = None
        args.vpt_weight_decay = None
        args.adapter_reduction = None
        args.adapter_lr = None
        args.adapter_weight_decay = None

        assert args.lora_rank > 0, f"Expected lora_rank to be greater than 0, got {args.lora_rank}"
        assert args.lora_alpha > 0, f"Expected lora_alpha to be greater than 0, got {args.lora_alpha}"
        assert 0 <= args.lora_dropout < 1, f"Expected lora_dropout to be between 0 and 1, got {args.lora_dropout}"
        assert args.lora_lr > 0, f"Expected lora_lr to be greater than 0, got {args.lora_lr}"
        assert args.lora_weight_decay > 0, f"Expected lora_weight_decay to be greater than or equal to 0, got {args.lora_weight_decay}"
    else:
        args.lora_rank = None
        args.lora_alpha = None
        args.lora_dropout = None
        args.lora_lr = None
        args.lora_weight_decay = None
    

    if "vit" not in args.model_name.lower():
        args.num_vpt = None
        args.vpt_drop = None
        args.vpt_lr = None
        args.vpt_weight_decay = None
    else:
        args.backbone_lr = None
        args.backbone_weight_decay = None
        
        if not (args.lora or args.adapter):  # Use VPT only if not using LoRA or adapter
            assert args.num_vpt > 0, f"Expected num_vpt to be greater than 0, got {args.num_vpt}"
            assert 0 <= args.vpt_drop < 1, f"Expected vpt_drop to be between 0 and 1, got {args.vpt_drop}"
            assert args.vpt_lr > 0, f"Expected vpt_lr to be greater than 0, got {args.vpt_lr}"
            assert args.vpt_weight_decay >= 0, f"Expected vpt_weight_decay to be greater than or equal to 0, got {args.vpt_weight_decay}"
        else:
            args.num_vpt = None
            args.vpt_drop = None
            args.vpt_lr = None
            args.vpt_weight_decay = None

    if args.reg_loss != "dm" and args.aux_loss != "dm":
        args.numItermax = None
        args.regularization = None
    
    if args.reg_loss != "msmae" and args.aux_loss != "msmae":
        args.scales = None
        args.min_scale_weight = None
        args.max_scale_weight = None
        args.alpha = None
    else:
        assert args.max_scale_weight >= args.min_scale_weight >= 0, f"Expected max_scale_weight to be greater than or equal to min_scale_weight, got {args.min_scale_weight} and {args.max_scale_weight}"
        assert 1 >= args.alpha > 0, f"Expected alpha to be between 0 and 1, got {args.alpha}"

    assert args.hdba_w >= 0, f"Expected hdba_w to be non-negative, got {args.hdba_w}"
    assert args.high_bin_min_count >= 0, f"Expected high_bin_min_count to be non-negative, got {args.high_bin_min_count}"
    assert 0 < args.hdba_block_q < 1, f"Expected hdba_block_q to be between 0 and 1, got {args.hdba_block_q}"
    assert args.hdba_margin >= 0, f"Expected hdba_margin to be non-negative, got {args.hdba_margin}"
    assert args.hdba_warmup_epochs >= 0, f"Expected hdba_warmup_epochs to be non-negative, got {args.hdba_warmup_epochs}"
    assert args.hdba_ramp_epochs >= 0, f"Expected hdba_ramp_epochs to be non-negative, got {args.hdba_ramp_epochs}"
    assert args.hdba_img_count_thr >= 0, f"Expected hdba_img_count_thr to be non-negative, got {args.hdba_img_count_thr}"
    assert 0 < args.hdba_top_ratio <= 1, f"Expected hdba_top_ratio to be in (0, 1], got {args.hdba_top_ratio}"
    assert args.hdba_block_count_thr >= 0, f"Expected hdba_block_count_thr to be non-negative, got {args.hdba_block_count_thr}"
    assert args.hdba_u_w >= 0, f"Expected hdba_u_w to be non-negative, got {args.hdba_u_w}"
    assert 0 <= args.pi_nonzero_thr <= 1, f"Expected pi_nonzero_thr to be between 0 and 1, got {args.pi_nonzero_thr}"
        
    if args.scheduler == "step":
        args.T_0 = None
        args.T_mult = None
        args.T_max = None
    elif args.scheduler == "cos":
        args.step_size = None
        args.gamma = None
        args.T_0 = None
        args.T_mult = None
    else:
        args.step_size = None
        args.gamma = None
        args.T_max = None
    
    args.nprocs = torch.cuda.device_count()
    args.batch_size = int(args.batch_size / args.nprocs)
    args.num_workers = int(args.num_workers / args.nprocs)

    if args.semi:
        assert args.label_percent < 100.0, "semi 模式下 label_percent 应该小于 100，例如 10。"
        assert args.nprocs == 1, "这份半监督最小版本先只支持单卡。请使用 CUDA_VISIBLE_DEVICES=0。"
        args.batch_size_u = int(args.batch_size_u / args.nprocs)

        if args.eval_freq < 1:
            print("semi 模式暂不建议使用 eval_freq < 1，自动改为 eval_freq=1。")
            args.eval_freq = 1
    else:
        args.batch_size_u = None
        args.ema_decay = None
        args.lambda_u_max = None
        args.lambda_u_min = None
        args.ssl_warmup_epochs = None
        args.ssl_rampup_epochs = None
        args.ssl_map_w = None
        args.ssl_count_w = None
    
    if args.ckpt_dir_name is None:
        hyperparams_dict = (vars(args)).copy()
        hyperparams_dict.pop("save_freq")
        hyperparams_dict.pop("save_best_k")
        hyperparams_dict.pop("local_rank")
        hyperparams_dict.pop("num_workers")
        hyperparams_dict.pop("nprocs")
        hyperparams_dict.pop("ckpt_dir_name")
        hyperparams_dict = json.dumps(hyperparams_dict, sort_keys=True)
        args.hash = hashlib.sha256(hyperparams_dict.encode("utf-8")).hexdigest()

        if "CLIP_" in args.model_name:
            ckpt_dir_name = f"{args.model_name}_{args.clip_weight_name}_"
            if "ViT" in args.model_name:
                ckpt_dir_name += f"{args.num_vpt}_{args.vpt_drop}_"
        else:
            ckpt_dir_name = f"{args.model_name}_{args.block_size}_"
        ckpt_dir_name += f"{args.weight_cls}+{args.weight_reg}x{(args.reg_loss)}+{args.weight_aux}{(args.aux_loss)}_"
        ckpt_dir_name += f"{args.optimizer}_{args.scheduler}_{args.hash[:8]}"
    
    else:
        ckpt_dir_name = args.ckpt_dir_name

    args.ckpt_dir = os.path.join(current_dir, "checkpoints", args.dataset, ckpt_dir_name)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print(f"Using {args.nprocs} GPUs.")
    if args.nprocs > 1:
        if args.in_memory_dataset:
            print("In-memory dataset is not supported for distributed training. Using disk-based dataset instead.")
            args.in_memory_dataset = False
        mp.spawn(run, nprocs=args.nprocs, args=(args.nprocs, args))
    else:
        run(0, 1, args)


if __name__ == "__main__":
    main()
