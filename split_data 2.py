import os
import shutil
import random

# Define the paths
test_folder = "./3D-FUTURE-scene/test"
val_folder = "./3D-FUTURE-scene/val"
idmap_folder = os.path.join(test_folder, "idmap")
image_folder = os.path.join(test_folder, "image")

val_idmap_folder = os.path.join(val_folder, "idmap")
val_image_folder = os.path.join(val_folder, "image")

# Create the val folders if they don't exist
# os.makedirs(val_idmap_folder, exist_ok=True)
# os.makedirs(val_image_folder, exist_ok=True)

# List all files in the test folders
idmap_files = sorted(os.listdir(val_idmap_folder))
image_files = sorted(os.listdir(val_image_folder))

# Remove file extensions for comparison
idmap_file_names = [os.path.splitext(f)[0] for f in idmap_files]
image_file_names = [os.path.splitext(f)[0] for f in image_files]

# Ensure the files match and print the differences if they don't
if idmap_file_names != image_file_names:
    idmap_set = set(idmap_file_names)
    image_set = set(image_file_names)
    in_idmap_not_image = idmap_set - image_set
    in_image_not_idmap = image_set - idmap_set
    print("Files in idmap not in image:", in_idmap_not_image)
    print("Files in image not in idmap:", in_image_not_idmap)
    assert idmap_file_names == image_file_names, "Files in idmap and image folders do not match!"

# # Shuffle the list of files and split into half
# random.seed(42)  # For reproducibility
# random.shuffle(idmap_files)
# split_index = len(idmap_files) // 2

# val_idmap_files = idmap_files[:split_index]
# val_image_files = [f.replace('.png', '.jpg') for f in val_idmap_files]  # Convert to image file names

# # Move the selected files to the val folder
# for file_name in val_idmap_files:
#     shutil.move(os.path.join(idmap_folder, file_name), os.path.join(val_idmap_folder, file_name))
# for file_name in val_image_files:
#     shutil.move(os.path.join(image_folder, file_name), os.path.join(val_image_folder, file_name))

# Print the results
print({
    "val_idmap_count": len(os.listdir(val_idmap_folder)),
    "val_image_count": len(os.listdir(val_image_folder)),
    "test_idmap_count": len(os.listdir(idmap_folder)),
    "test_image_count": len(os.listdir(image_folder)),
})

# import os
# import shutil

# # Define the paths
# test_folder = "./3D-FUTURE-scene/val"
# idmap_folder = os.path.join(test_folder, "idmap")
# image_folder = os.path.join(test_folder, "image")

# # Function to move .DS_Store files
# def move_ds_store(folder, destination):
#     ds_store_path = os.path.join(folder, '.DS_Store')
#     if os.path.exists(ds_store_path):
#         shutil.move(ds_store_path, os.path.join(destination, f'.DS_Store_{os.path.basename(folder)}'))
#         print(f"Moved: {ds_store_path} to {destination}")
#     else:
#         print(f"No .DS_Store file found in {folder}")

# # Move .DS_Store files to the parent test folder
# move_ds_store(idmap_folder, ".")
# move_ds_store(image_folder, ".")