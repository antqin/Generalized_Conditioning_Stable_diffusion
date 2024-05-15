import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline

device = "cuda"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe = pipe.to(device)

url = "https://raw.githubusercontent.com/3D-FRONT-FUTURE/3D-FUTURE-ToolBox/master/data/scene_data/images/raw_scene.png"

response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((768, 512))

prompt = "A cozy living room with large windows, blue drapes, dark wood floor, yellow rug, vintage yellow sofa with plaid cushions, modern coffee table, dark armchair, bar stool, white dresser, floral painting, and pink-gold chandelier."

images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
images[0].save("fantasy_landscape.png")