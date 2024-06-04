import torch
from diffusers import StableDiffusionPipeline

# Load the pretrained stable diffusion model
model_name_or_path = "stabilityai/stable-diffusion-2-1"
pipeline = StableDiffusionPipeline.from_pretrained(model_name_or_path, torch_dtype=torch.float16).to("cuda")

# Load the LoRA weights
lora_weights_path = "./finetune-lora-output/checkpoint-900/model.safetensors"
pipeline.unet.load_attn_procs(lora_weights_path)

# Define the prompt for image generation
prompt = "generate a realistic interior room design"

# Generate and save images with different numbers of inference steps
for steps in [5, 10, 25, 50, 75, 100]:
    generator = torch.Generator(device="cuda").manual_seed(42)  # Set seed for reproducibility
    images = pipeline(prompt, num_inference_steps=steps, generator=generator).images
    images[0].save(f"generated_image_{steps}_steps.png")
