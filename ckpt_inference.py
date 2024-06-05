import torch
from diffusers import StableDiffusionPipeline

# Define paths
model_ckpt_path = "./finetune-lora-dreambooth-output/dreambooth-model.ckpt"
output_dir = "./ckpt_text2img_output"

# Set seed for reproducibility
seed = 1337
torch.manual_seed(seed)

# Load the base model
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float32)

# Load the .ckpt weights
pipe.load_lora_weights(model_ckpt_path)

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
