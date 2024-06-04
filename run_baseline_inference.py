import torch
from diffusers import StableDiffusionPipeline

# Load the pretrained stable diffusion model
model_name_or_path = "stabilityai/stable-diffusion-2-1"
pipeline = StableDiffusionPipeline.from_pretrained(model_name_or_path, torch_dtype=torch.float16).to("cuda")

# Define the prompt for image generation
prompt = "generate a realistic interior room design of a dining room"

# Generate and save images with different numbers of inference steps for comparison
for steps in [5, 10, 25, 50, 75, 100]:
    generator = torch.Generator(device="cuda").manual_seed(1337)  # Set seed for reproducibility
    images = pipeline(prompt, num_inference_steps=steps, generator=generator).images
    images[0].save(f"baseline_dining_{steps}_steps.png")
