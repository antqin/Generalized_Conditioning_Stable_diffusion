import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer, StableDiffusionPipeline, LoRAConfig, LoRATrainer
from torchvision import transforms
import torch
import torch.nn as nn

class InteriorDesignDataset(Dataset):
    def __init__(self, idmap_dir, image_dir, transform=None):
        self.idmap_dir = idmap_dir
        self.image_dir = image_dir
        self.transform = transform
        self.image_pairs = self._load_image_pairs()

    def _load_image_pairs(self):
        image_pairs = []
        for file_name in os.listdir(self.idmap_dir):
            if file_name.endswith(".png"):
                base_name = file_name[:-4]
                idmap_path = os.path.join(self.idmap_dir, file_name)
                image_path = os.path.join(self.image_dir, base_name + ".jpg")
                if os.path.exists(image_path):
                    image_pairs.append((idmap_path, image_path))
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        idmap_path, image_path = self.image_pairs[idx]
        idmap_image = Image.open(idmap_path).convert("RGB")
        goal_image = Image.open(image_path).convert("RGB")
        if self.transform:
            idmap_image = self.transform(idmap_image)
            goal_image = self.transform(goal_image)
        return idmap_image, goal_image

# Set up directories
train_idmap_dir = "3D-FUTURE-scene/train/idmap"
train_image_dir = "3D-FUTURE-scene/train/image"
val_idmap_dir = "3D-FUTURE-scene/val/idmap"
val_image_dir = "3D-FUTURE-scene/val/image"

# Define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Create datasets
train_dataset = InteriorDesignDataset(train_idmap_dir, train_image_dir, transform=transform)
val_dataset = InteriorDesignDataset(val_idmap_dir, val_image_dir, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Initialize the tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Initialize the Stable Diffusion model
model = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")

# Define LoRA configuration
lora_config = LoRAConfig(
    r=4,
    alpha=32,
    dropout=0.1
)

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
def train_model(train_loader, val_loader, model, tokenizer, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        for i, (idmap_images, goal_images) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Tokenize the prompt
            inputs = tokenizer(["generate a realistic interior room design"] * idmap_images.size(0), return_tensors="pt", padding=True).input_ids
            
            # Forward pass
            outputs = model(idmap_images, inputs)
            
            # Compute loss
            loss = criterion(outputs, goal_images)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for idmap_images, goal_images in val_loader:
                inputs = tokenizer(["generate a realistic interior room design"] * idmap_images.size(0), return_tensors="pt", padding=True).input_ids
                outputs = model(idmap_images, inputs)
                val_loss += criterion(outputs, goal_images).item()
            val_loss /= len(val_loader)
        print(f"Validation Loss after Epoch {epoch+1}: {val_loss:.4f}")

# Train the model
train_model(train_loader, val_loader, model, tokenizer, criterion, optimizer, num_epochs=5)

# Save the fine-tuned model
model.save_pretrained("fine_tuned_stable_diffusion")
