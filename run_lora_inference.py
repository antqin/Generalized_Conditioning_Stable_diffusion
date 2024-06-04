import torch
from diffusers import StableDiffusionPipeline

# Load the pretrained stable diffusion model
model_name_or_path = "stabilityai/stable-diffusion-2-1"
pipeline = StableDiffusionPipeline.from_pretrained(model_name_or_path, torch_dtype=torch.float16).to("cuda")

# Load the LoRA weights
lora_weights_path = "./finetune-lora-output/checkpoint-900/pytorch_lora_weights.safetensors"
pipeline.unet.load_attn_procs(lora_weights_path)

# Define the prompt for image generation
prompt = "generate a realistic interior room design of a modern living room with a TV, coffee table, chair, and a view of the backyard"

# Generate the image
generator = torch.Generator(device="cuda").manual_seed(42)  # Set seed for reproducibility
images = pipeline(prompt, num_inference_steps=50, generator=generator).images

# Save the generated image
images[0].save("generated_living_room_image.png")
