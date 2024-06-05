import torch
from diffusers import StableDiffusionPipeline

# Define paths
model_name = "stabilityai/stable-diffusion-2-1"
lora_weights_path = "./finetune-lora-output/pytorch_lora_weights.safetensors"
output_dir = "./finetune-lora-output"

# Set seed for reproducibility
seed = 1337
torch.manual_seed(seed)

# Load the base model
pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)

# Load the LoRA weights
pipe.load_lora_weights(lora_weights_path)

# Move the pipeline to the GPU (if available)
pipe = pipe.to("cuda")

# Define the prompt
prompt = "generate a realistic interior room design of a dining room"

# Generate images
generator = torch.Generator("cuda").manual_seed(seed)
for steps in [5, 10, 25, 50, 75, 100]:
    with torch.autocast("cuda"):
        image = pipe(prompt, num_inference_steps=steps, guidance_scale=7.5, generator=generator).images[0]
        image.save(f"{output_dir}/generated_dining_{steps}.png")
        print("Image saved to", f"{output_dir}/generated_dining_{steps}.png")
