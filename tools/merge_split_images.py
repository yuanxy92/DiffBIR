import cv2
import numpy as np
from PIL import Image
import os

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

# Usage example
block_folder = 'C:/Projects/Code/DiffBIR/results/LBAS_0708'
output_image_path = 'C:/Projects/Code/DiffBIR/results/LBAS_0708.png'
canvas_img_filename = 'C:/Projects/Code/DiffBIR/inputs/LBAS/00000.png'
canvas_image = cv2.imread(canvas_img_filename, cv2.IMREAD_COLOR)

image_size = (1000, 1000)  # Size of the original image
block_size = (400, 400)  # Size of each block
overlap = 200  # Overlap in pixels
scale = 2

canvas_image = cv2.resize(canvas_image, image_size)
merge_blocks_with_poisson_blending(block_folder, output_image_path, canvas_image, image_size, block_size, overlap, scale)
