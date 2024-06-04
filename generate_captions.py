import os
import json

# Define the path to the images directory and the output metadata file
images_dir = './3D-FUTURE-scene/lora-train/image'
metadata_file = os.path.join(images_dir, 'metadata.jsonl')

# Define the caption for each image
caption = "generate a realistic interior room design"

# Get a list of all image files in the directory
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Create the metadata entries
metadata_entries = []
for image_file in image_files:
    entry = {
        "file_name": image_file,
        "text": caption
    }
    metadata_entries.append(entry)

# Write the metadata entries to the metadata.jsonl file
with open(metadata_file, 'w') as f:
    for entry in metadata_entries:
        f.write(json.dumps(entry) + '\n')

print(f"Metadata file created with {len(metadata_entries)} entries.")
