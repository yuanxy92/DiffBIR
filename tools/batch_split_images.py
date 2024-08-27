from PIL import Image
import os

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

# Usage example
image_path = 'C:/Projects/Code/DiffBIR/inputs/LBAS/00000.png'
output_folder = 'C:/Projects/Code/DiffBIR/inputs/LBAS/0708'
block_size = (200, 200)  # Width, Height of each block
overlap = 100  # Overlap in pixels

split_image_into_blocks(image_path, output_folder, block_size, overlap)
