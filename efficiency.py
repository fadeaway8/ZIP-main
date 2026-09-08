from argparse import ArgumentParser
import time
import os
import torch
import torchvision.transforms as transforms
from contextlib import nullcontext
import json
from models import get_model


parser = ArgumentParser(description="Train an EBC model.")
parser.add_argument("--model_info_path", type=str, required=True, help="Path to the model information file.")

parser.add_argument("--batch_size", type=int, default=1, help="Batch size for the model.")
parser.add_argument("--height", type=int, default=768, help="Height of the input image.")
parser.add_argument("--width", type=int, default=1024, help="Width of the input image.")

parser.add_argument("--num_iterations", type=int, default=200, help="Number of iterations to run the model.")
parser.add_argument("--num_warmup", type=int, default=20, help="Dispose of the first N iterations.")

parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], help="Device to run the model on. Options are 'cpu', 'cuda', or 'mps'.")
parser.add_argument("--amp", action="store_true", help="Enable autocast mixed precision (fp16/bf16).")
parser.add_argument("--half", action="store_true", help="Use half precision for the model.")
parser.add_argument("--channels_last", action="store_true", help="Use NHWC memory format (recommended for CUDA).")
parser.add_argument("--compile", action="store_true", help="Enable torch.compile if available.")
parser.add_argument("--threads", type=int, default=None, help="torch.set_num_threads(threads) for CPU")
parser.add_argument("--sleep_time", type=float, default=0.0, help="Seconds to sleep after *each* iteration (cool-down).")

_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def _dummy_input(bs, h, w, device, half, channels_last):
    x = torch.rand(bs, 3, h, w, device=device)
    x = _normalize(x)
    if half:
        x = x.half()
    if channels_last:
        x = x.to(memory_format=torch.channels_last)
    return x


def _maybe_sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def benchmark(
    model: torch.nn.Module,
    inp: torch.Tensor,
    warmup: int,
    steps: int,
    amp: bool,
    sleep_time: float = 0.0
):
    cm = torch.autocast(device_type=inp.device.type) if amp else nullcontext()

    # --- warm-up ---
    for _ in range(warmup):
        with cm:
            _ = model(inp)
    _maybe_sync(inp.device)

    # --- timed loop ---
    total_time = 0.0
    for _ in range(steps):
        tic = time.perf_counter()
        with cm:
            _ = model(inp)
        
        toc = time.perf_counter()
        total_time += toc - tic

        if sleep_time > 0:
            time.sleep(sleep_time)
    
    _maybe_sync(inp.device)

    fps = steps / total_time
    return fps, total_time / steps


def main(args):
    assert os.path.isfile(args.model_info_path), \
        f"{args.model_info_path} not found"

    model = get_model(model_info_path=args.model_info_path)
    model.eval()

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.half:
        model = model.half()

    device = torch.device(args.device)
    model = model.to(device)

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")

    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)

    inp = _dummy_input(
        args.batch_size,
        args.height, 
        args.width,
        device,
        args.half,
        args.channels_last
    )

    fps, t_avg = benchmark(
        model,
        inp,
        warmup=args.num_warmup,
        steps=args.num_iterations,
        amp=args.amp,
        sleep_time=args.sleep_time
    )

    cfg = vars(args)
    cfg.pop("model_info_path")
    print(json.dumps(cfg, indent=2))
    print(f"\nAverage latency: {t_avg*1000:6.2f} ms  |  FPS: {fps:,.2f}")


if __name__ == "__main__":
    main(parser.parse_args())


# CUDA @FP16 + channels_last + torch.compile
# python efficiency.py \
#   --model_info_path checkpoints/shb/ebc_p/best_mae.pth \
#   --device cuda --half --amp --channels_last --compile

# CUDA @AMP + channels_last + torch.compile
# python efficiency.py \
#   --model_info_path checkpoints/shb/ebc_p/best_mae.pth \
#   --device cuda --amp --channels_last --compile

# CUDA @FP32 + channels_last + torch.compile
# python efficiency.py \
#   --model_info_path checkpoints/shb/ebc_p/best_mae.pth \
#   --device cuda --channels_last --compile

# AMD 5900X (12 Core) + channels_last + torch.compile
# export OMP_NUM_THREADS=12; export MKL_NUM_THREADS=12
# python efficiency.py \
#   --model_info_path checkpoints/shb/ebc_p/best_mae.pth \
#   --device cpu --threads 12 --channels_last --compile

# Apple M1 Pro (6 Performance Cores). Compiling makes it slower.
# export OMP_NUM_THREADS=6; export VECLIB_MAXIMUM_THREADS=6
# python efficiency.py \
#   --model_info_path checkpoints/shb/ebc_p/best_mae.pth \
#   --device cpu --threads 6

# Apple M1 Pro MPS @FP32 + torch.compile
# python efficiency.py \
#   --model_info_path checkpoints/shb/ebc_p/best_mae.pth \
#   --device mps --channels_last --compile