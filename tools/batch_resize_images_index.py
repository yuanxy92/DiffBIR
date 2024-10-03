from PIL import Image
import os
import os.path

# Define the path to the folder containing the images
root_dir = '/data/xiaoyun/OV6946_Arm_6_cameras/20240829'
out_dir = '/data/xiaoyun/OV6946_Arm_6_cameras/20240829_256px_v2'
datanames = ['data_0829_3', 'data_0829_4', 'data_0829_5',
             'data_0829_6', 'data_0829_7', 'data_0829_8']
camera_indices = ['0', '1', '2', '4', '5', '6']
start = 70
stop = 900
step = 1

for data_idx in range(len(datanames)):
    input_folder = f'{root_dir}/{datanames[data_idx]}_undis'
    output_folder = f'{out_dir}/{datanames[data_idx]}_undis256'
    # Define the desired size (width, height)
    new_size = (256, 256)
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for cam_idx in camera_indices:
        # Check if the file is an image
        for img_idx in range(start, stop + 1, step):
            # Open an image file
            basename = f'camera_{cam_idx}_frame_{img_idx}_corrected.png'
            filename = f'{input_folder}/{basename}'
            if os.path.isfile(filename):
                with Image.open(filename) as img:
                    # Resize the image
                    img_resized = img.resize(new_size, Image.LANCZOS)
                    # Save the image to the output folder
                    img_resized.save(os.path.join(output_folder, basename))

    print("All images have been resized and saved to", output_folder)