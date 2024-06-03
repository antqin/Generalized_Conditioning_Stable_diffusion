from diffusers import StableDiffusionPipeline

# Path to your checkpoint directory
checkpoint_dir = "output/final_checkpoint"

# Load the model
pipeline = StableDiffusionPipeline.from_pretrained(checkpoint_dir)

# Move the model to GPU (if available)
pipeline.to("cuda")

# Now you can use the pipeline for inference
prompt = "generate a realistic interior room design"
image = pipeline(prompt).images[0]

# Save or display the generated image
image.save("generated_image.png")
