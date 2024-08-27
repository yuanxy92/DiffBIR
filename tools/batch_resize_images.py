from PIL import Image
import os

# Define the path to the folder containing the images
input_folder = 'C:/Projects/Code/DiffBIR/inputs/OV6946/0823_400'
output_folder = 'C:/Projects/Code/DiffBIR/inputs/OV6946/0823_200'

# Define the desired size (width, height)
new_size = (200, 200)

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    # Check if the file is an image
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
        # Open an image file
        with Image.open(os.path.join(input_folder, filename)) as img:
            # Resize the image
            img_resized = img.resize(new_size, Image.LANCZOS)
            img_rotated = img_resized.rotate(90, expand=True)
            # Save the image to the output folder
            img_rotated.save(os.path.join(output_folder, filename))

print("All images have been resized and saved to", output_folder)