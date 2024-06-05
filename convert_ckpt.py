import torch
from diffusers import StableDiffusionPipeline

# Define paths
model_name = "stabilityai/stable-diffusion-2-1"
lora_weights_path = "./finetune-lora-dreambooth-output/pytorch_lora_weights.safetensors"
ckpt_output_path = "./finetune-lora-dreambooth-output/pytorch_lora_weights.ckpt"

# Load the base model
pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)

# Load the LoRA weights
pipe.load_lora_weights(lora_weights_path)

# Extract the state dict
state_dict = pipe.unet.state_dict()

# Save the state dict in .ckpt format
torch.save(state_dict, ckpt_output_path)

print(f"Model weights saved to {ckpt_output_path}")