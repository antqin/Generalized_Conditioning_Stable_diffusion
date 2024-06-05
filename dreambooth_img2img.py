import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os

# Define paths
model_name = "stabilityai/stable-diffusion-2-1"
lora_weights_path = "./finetune-lora-dreambooth-output/pytorch_lora_weights.safetensors"
# Uncomment below line for LoRA without Dreambooth
# lora_weights_path = "./finetune-lora-output/pytorch_lora_weights.safetensors"
input_dir = "./baseline_generations_test"
output_dir = "./finetune-lora-dreambooth-output/img2img_outputs"
# Uncomment below line for LoRA without Dreambooth
# output_dir = "./finetune-lora-dreambooth-output/img2img_outputs"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Set seed for reproducibility
seed = 1337
torch.manual_seed(seed)

# Load the base model
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_name, torch_dtype=torch.float32)

# Load the LoRA weights
pipe.load_lora_weights(lora_weights_path)

# Move the pipeline to the GPU (if available)
pipe = pipe.to("cuda")

# Define the prompt
prompt = "generate a realistic interior room design that follows the layout of the provided idmap image but uses realistic colors. do not use the same colors as the idmap image"

# Set generator for reproducibility
generator = torch.Generator("cuda").manual_seed(seed)

# Iterate over images in the input directory and perform img2img inference
for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        input_path = os.path.join(input_dir, filename)
        image = Image.open(input_path).convert("RGB")
        image = image.resize((512,512))
        for steps in [100]:
            with torch.autocast("cuda"):
                generated_image = pipe(prompt=prompt, image=image, strength=1, num_inference_steps=100, guidance_scale=15).images[0]
                #generated_image = pipe(prompt=prompt, init_image=image, strength=0.75, num_inference_steps=steps, guidance_scale=7.5, generator=generator).images[0]
                output_path = os.path.join(output_dir, f"{filename.split('.')[0]}_steps_{steps}.png")
                generated_image.save(output_path)
                print("Image saved to", output_path)
