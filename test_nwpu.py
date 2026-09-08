import torch
import torch.nn.functional as F
from argparse import ArgumentParser
import os
from tqdm import tqdm

current_dir = os.path.abspath(os.path.dirname(__file__))

from datasets import NWPUTest, Resize2Multiple
from models import get_model
from utils import get_config, sliding_window_predict

parser = ArgumentParser(description="Generate the test result of a trained model on the NWPU-Crowd test set.")
# Parameters for model
parser.add_argument("--weight_path", type=str, required=True, help="The directory to the checkpoints. This should also include the model_info.pkl file.")
parser.add_argument("--output_filename", type=str, default="test_results", help="The name of the output file.")

# Parameters for evaluation
parser.add_argument("--input_size", type=int, default=224, help="The size of the input image.")
parser.add_argument("--sliding_window", action="store_true", help="Use sliding window strategy for evaluation.")
parser.add_argument("--max_input_size", type=int, default=4096, help="The maximum size of the input image in evaluation. Images larger than this will be processed using sliding window by force to avoid OOM.")
parser.add_argument("--max_num_windows", type=int, default=8, help="The maximum number of windows to be simultaneously processed.")
parser.add_argument("--resize_to_multiple", action="store_true", help="Resize the image to the nearest multiple of the input size.")
parser.add_argument("--stride", type=int, default=None, help="The stride for sliding window strategy.")
parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision for evaluation.")
parser.add_argument("--device", type=str, default="cuda", help="The device to use for evaluation.")


def main(args: ArgumentParser):
    print("Testing a trained model on the NWPU-Crowd test set.")
    device = torch.device(args.device)
    _ = get_config(vars(args).copy(), mute=False)

    model = get_model(model_info_path=args.weight_path).to(device)
    model.eval()

    sliding_window = args.sliding_window
    if args.resize_to_multiple:
        transforms = Resize2Multiple(base=args.input_size)
    else:
        transforms = None

    dataset = NWPUTest(transforms=transforms, return_filename=True)

    image_ids = []
    preds = []
    input_size = args.input_size

    for idx in tqdm(range(len(dataset)), desc="Testing on NWPU"):
        image, image_path = dataset[idx]
        image = image.unsqueeze(0)  # add batch dimension
        image = image.to(device)  # add batch dimension
        image_height, image_width = image.shape[-2:]

        # Resize image if it's smaller than the window size
        aspect_ratio = image_width / image_height
        if image_height < input_size:
            new_height = input_size
            new_width = int(new_height * aspect_ratio)
            image = F.interpolate(image, size=(new_height, new_width), mode="bicubic", align_corners=False)
            image_height, image_width = new_height, new_width
        if image_width < input_size:
            new_width = input_size
            new_height = int(new_width / aspect_ratio)
            image = F.interpolate(image, size=(new_height, new_width), mode="bicubic", align_corners=False)
            image_height, image_width = new_height, new_width

        with torch.set_grad_enabled(False), torch.autocast(device_type="cuda", enabled=args.amp):
            if sliding_window or (args.max_input_size is not None and (image_height * image_width) > args.max_input_size ** 2):
                pred_den_map = sliding_window_predict(model, image, input_size, args.stride, args.max_num_windows)
            else:
                pred_den_map = model(image)

            pred_count = pred_den_map.sum(dim=(1, 2, 3)).item()

        image_ids.append(os.path.basename(image_path).split(".")[0])
        preds.append(pred_count)

    result_dir = os.path.join(current_dir, "nwpu_test_results")
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, f"{args.output_filename}.txt"), "w") as f:
        for idx, (image_id, pred) in enumerate(zip(image_ids, preds)):
            if idx != len(image_ids) - 1:
                f.write(f"{image_id} {pred}\n")
            else:
                f.write(f"{image_id} {pred}")  # no newline at the end of the file


if __name__ == "__main__":
    args = parser.parse_args()
    # Sliding window prediction will be used if args.sliding_window is True, or when the image size is larger than args.max_input_size
    args.stride = args.stride or args.input_size
    assert os.path.exists(args.weight_path), f"Checkpoint directory {args.weight_path} does not exist."
    main(args)
