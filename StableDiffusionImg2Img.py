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

prompt = "A cozy living room with large windows featuring sheer white curtains and blue drapes with golden tiebacks, a dark wood floor, a light yellow rug, and a vintage-style yellow sofa with plaid cushions. The room includes a modern coffee table, a dark armchair, a high brown bar stool, a small white dresser, a colorful floral painting on the wall, and a pink and gold chandelier hanging from the ceiling."

images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
images[0].save("fantasy_landscape.png")