import os
import random
import shutil

# Define source and destination directories
source_image_dir = './3D-FUTURE-scene/train/image'
source_idmap_dir = './3D-FUTURE-scene/train/idmap'
dest_image_dir = './3D-FUTURE-scene/lora-train/image'
dest_idmap_dir = './3D-FUTURE-scene/lora-train/idmap'

# Create destination directories if they don't exist
os.makedirs(dest_image_dir, exist_ok=True)
os.makedirs(dest_idmap_dir, exist_ok=True)

# Get list of all image files in the source image directory
image_files = [f for f in os.listdir(source_image_dir) if f.endswith('.jpg')]

# Randomly select 30 images
selected_images = random.sample(image_files, 30)

# Copy selected images and their corresponding idmaps to the destination directory
for image_file in selected_images:
    base_name = image_file[:-4]
    idmap_file = base_name + '.png'

    # Copy image file
    shutil.copy(os.path.join(source_image_dir, image_file), os.path.join(dest_image_dir, image_file))

    # Copy idmap file
    shutil.copy(os.path.join(source_idmap_dir, idmap_file), os.path.join(dest_idmap_dir, idmap_file))

print(f"Selected 30 images and their corresponding idmaps have been copied to {dest_image_dir} and {dest_idmap_dir}")
