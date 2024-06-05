import torch
from diffusers import StableDiffusionPipeline

# Define paths
model_ckpt_path = "./finetune-lora-dreambooth-output/dreambooth-model.ckpt"
output_dir = "./ckpt-dreambooth-output"

# Set seed for reproducibility
seed = 1337
torch.manual_seed(seed)

# Load the model checkpoint
checkpoint = torch.load(model_ckpt_path, map_location="cuda")

# Load the base model
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float32)

# Load the state dict into the model
pipe.load_state_dict(checkpoint['state_dict'])

# Move the pipeline to the GPU (if available)
pipe = pipe.to("cuda")

# Define the prompt
prompt = "generate a realistic interior room design"

# Generate images
generator = torch.Generator("cuda").manual_seed(seed)
for steps in [100]:
    with torch.autocast("cuda"):
        image = pipe(prompt, num_inference_steps=steps, guidance_scale=7.5, generator=generator).images[0]
        image.save(f"{output_dir}/generated_{steps}.png")
        print("Image saved to", f"{output_dir}/generated_{steps}.png")

