import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import load_file as safe_load

# Path to your checkpoint directory
checkpoint_dir = "output/final_checkpoint"

# Load the tokenizer
tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="tokenizer")

# Load the models
text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae")
unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="unet")

# Load weights from safetensors
text_encoder_weights = safe_load(f"{checkpoint_dir}/model_1.safetensors", device="cpu")
vae_weights = safe_load(f"{checkpoint_dir}/model_1.safetensors", device="cpu")
unet_weights = safe_load(f"{checkpoint_dir}/model.safetensors", device="cpu")

# Load state dictionaries
text_encoder.load_state_dict(text_encoder_weights)
vae.load_state_dict(vae_weights)
unet.load_state_dict(unet_weights)

# Create the pipeline
pipeline = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
)

# Move the pipeline to GPU (if available)
pipeline.to("cuda")

# Now you can use the pipeline for inference
prompt = "generate a realistic interior room design"
image = pipeline(prompt).images[0]

# Save or display the generated image
image.save("generated_image.png")
