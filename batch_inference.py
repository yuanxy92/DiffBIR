import os
from argparse import ArgumentParser, Namespace
from PIL import Image
import torch
import numpy as np
import cv2

from accelerate.utils import set_seed
from utils.inference import (
    V1InferenceLoop,
    BSRInferenceLoop, BFRInferenceLoop, BIDInferenceLoop, UnAlignedBFRInferenceLoop
)

def check_device(device: str) -> str:
    if device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA not available because the current PyTorch install was not "
                  "built with CUDA enabled.")
            device = "cpu"
    else:
        if device == "mps":
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print("MPS not available because the current PyTorch install was not "
                          "built with MPS enabled.")
                    device = "cpu"
                else:
                    print("MPS not available because the current MacOS version is not 12.3+ "
                          "and/or you do not have an MPS-enabled device on this machine.")
                    device = "cpu"
    print(f"using device {device}")
    return device

def parse_args() -> Namespace:
    parser = ArgumentParser()
    ### model parameters
    parser.add_argument("--task", type=str, choices=["sr", "dn", "fr", "fr_bg"])
    parser.add_argument("--upscale", type=float)
    parser.add_argument("--version", type=str, default="v2", choices=["v1", "v2"])
    ### sampling parameters
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--better_start", action="store_true")
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--tile_stride", type=int, default=256)
    parser.add_argument("--pos_prompt", type=str, default="")
    parser.add_argument("--neg_prompt", type=str, default="low quality, blurry, low-resolution, noisy, unsharp, weird textures")
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    ### input parameters
    parser.add_argument("--input", type=str)
    parser.add_argument("--n_samples", type=int, default=1)
    ### guidance parameters
    parser.add_argument("--guidance", action="store_true")
    parser.add_argument("--g_loss", type=str, default="w_mse", choices=["mse", "w_mse"])
    parser.add_argument("--g_scale", type=float, default=0.0)
    parser.add_argument("--g_start", type=int, default=1001)
    parser.add_argument("--g_stop", type=int, default=-1)
    parser.add_argument("--g_space", type=str, default="latent")
    parser.add_argument("--g_repeat", type=int, default=1)
    ### output parameters
    parser.add_argument("--output", type=str)
    ### common parameters
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    return parser.parse_args(args=[])

def split_image_into_blocks(image_path, output_folder, block_size=(100, 100), overlap=10):
    # Open the image file
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Calculate the step size (block size minus overlap)
    step_x = block_size[0] - overlap
    step_y = block_size[1] - overlap

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    block_count = 0

    # Loop through the image and extract blocks
    for y in range(0, img_height - block_size[1] + 1, step_y):
        for x in range(0, img_width - block_size[0] + 1, step_x):
            # Define the box to extract the block
            box = (x, y, x + block_size[0], y + block_size[1])
            block = img.crop(box)
            
            # Save the block with filename including x and y index
            block_filename = f'block_{x}_{y}.png'
            block.save(os.path.join(output_folder, block_filename))
            block_count += 1

    print(f"Total {block_count} blocks created and saved to {output_folder}") 

def merge_blocks_with_poisson_blending(block_folder, output_image_path, canvas, image_size, block_size=(100, 100), overlap=10, scale=2):
    # Create an empty canvas for the output image
    img_width, img_height = image_size
    count = np.zeros((img_height, img_width, 3), dtype=np.float32)
    canvas = canvas.astype(np.float32) / 255.0
    canvas_orig = np.copy(canvas)
    
    # Get list of block filenames
    block_filenames = sorted([f for f in os.listdir(block_folder) if f.endswith('.png')])

    for block_filename in block_filenames:
        # Extract block position from filename
        parts = block_filename.split('_')
        block_x = int(parts[1]) * scale
        block_y = int(parts[2].split('.')[0]) * scale
        
        # Load block image
        block_path = os.path.join(block_folder, block_filename)
        block_img = cv2.imread(block_path, cv2.IMREAD_COLOR)
        block_img = block_img.astype(np.float32) / 255.0  # Normalize

        # Define the region in the canvas
        start_x = block_x
        start_y = block_y
        end_x = start_x + block_size[0]
        end_y = start_y + block_size[1]

        # Poisson blending using `cv2.seamlessClone` needs adjustment here
        mask = np.ones_like(block_img)
        mask = mask.astype(np.float32)
        possion_blend_block = cv2.seamlessClone((block_img * 255).astype(np.uint8), 
            (canvas[start_y:end_y, start_x:end_x] * 255).astype(np.uint8), 
            (mask * 255).astype(np.uint8), (block_size[0] // 2, block_size[1] // 2), 
            cv2.NORMAL_CLONE)
        possion_blend_block = possion_blend_block.astype(np.float32) / 255
        
        canvas[start_y:end_y, start_x:end_x] = possion_blend_block

    canvas = np.clip(canvas * 255, 0, 255).astype(np.uint8)

    # Save the result
    cv2.imwrite(output_image_path, canvas)
    print(f"Image merged and saved to {output_image_path}")

def is_image_file(filename):
    # Check if the file is an image based on its extension
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    return filename.lower().endswith(tuple(valid_extensions))


def main(input_dir, output_dir, blksize = 200, overlap = 100, scale = 2):
    os.makedirs(output_dir, exist_ok=True)

    args = parse_args()
    args.task = 'sr'
    args.upscale = 2
    args.cfg_scale = 2.0
    args.device = check_device(args.device)
    set_seed(args.seed)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if is_image_file(file):
                image_filename = os.path.join(root, file)
                output_image_filename = os.path.join(output_dir, file)
                img = cv2.imread(image_filename)
                img_ds = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
                args.input = './results/temp_input'
                args.output = './results/temp_output'
                image_ds_path = './results/temp_img_ds.png'
                image_path = './results/temp_img_ds.png'
                cv2.imwrite(image_ds_path, img_ds)
                cv2.imwrite(image_path, img)

                # split images
                split_image_into_blocks(image_ds_path, args.input, block_size=(blksize, blksize), overlap=overlap)
                print(f'Split image of {image_filename} done!')

                # apply superresolution
                if args.version == "v1":
                    V1InferenceLoop(args).run()
                else:
                    supported_tasks = {
                        "sr": BSRInferenceLoop,
                        "dn": BIDInferenceLoop,
                        "fr": BFRInferenceLoop,
                        "fr_bg": UnAlignedBFRInferenceLoop
                    }
                    supported_tasks[args.task](args).run()
                    print("done!")
                print(f'Super-resolution image of {image_filename} done!')

                # merge images
                out_image_size = (img.shape[1], img.shape[0])
                merge_blocks_with_poisson_blending(args.output, output_image_filename, img, out_image_size, 
                    block_size=(blksize * scale, blksize * scale), overlap=overlap * scale, scale=scale)
                print(f'Merge image blocks of {image_filename} done!')


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"]="7"
    input_dir = '/data/xiaoyun/lbas_20240827/3dgs_output'
    output_dir = '/data/xiaoyun/lbas_20240827/3dgs_output_sr'
    main(input_dir, output_dir)