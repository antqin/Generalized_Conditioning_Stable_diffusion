import argparse
import hashlib
import itertools
import math
import os
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from lora_diffusion import inject_trainable_lora, save_lora_weight, extract_lora_ups_down
from torchvision import transforms

logger = get_logger(__name__)

class InteriorDesignDataset(Dataset):
    def __init__(self, idmap_dir, image_dir, tokenizer, size=512, center_crop=False):
        self.idmap_dir = idmap_dir
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop
        self.image_pairs = self._load_image_pairs()

    def _load_image_pairs(self):
        image_pairs = []
        for file_name in os.listdir(self.idmap_dir):
            if file_name.endswith(".png"):  # Ensure it is an image file
                base_name = file_name[:-4]
                idmap_path = os.path.join(self.idmap_dir, file_name)
                image_path = os.path.join(self.image_dir, base_name + ".jpg")
                if os.path.exists(image_path):
                    image_pairs.append((idmap_path, image_path))
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        idmap_path, image_path = self.image_pairs[idx]
        idmap_image = Image.open(idmap_path).convert("RGB")
        goal_image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.CenterCrop(self.size) if self.center_crop else transforms.RandomCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        idmap_image = transform(idmap_image)
        goal_image = transform(goal_image)

        example = {
            "instance_images": idmap_image,
            "instance_prompt_ids": self.tokenizer(
                "generate a realistic interior room design",
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids
        }
        return example

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune Stable Diffusion with LoRA")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="fp16")
    return parser.parse_args()

def main(args):
    logging_dir = os.path.join(args.output_dir, "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    if args.seed is not None:
        set_seed(args.seed)

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    unet.requires_grad_(False)
    unet_lora_params, _ = inject_trainable_lora(unet, r=args.lora_rank)

    text_encoder.requires_grad_(False)
    text_encoder_lora_params, _ = inject_trainable_lora(
        text_encoder,
        target_replace_module=["CLIPAttention"],
        r=args.lora_rank,
    )

    train_dataset = InteriorDesignDataset(
        idmap_dir=os.path.join(args.instance_data_dir, "idmap"),
        image_dir=os.path.join(args.instance_data_dir, "image"),
        tokenizer=tokenizer,
        size=args.resolution,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=1,
    )

    optimizer = torch.optim.AdamW(
        itertools.chain(*unet_lora_params, *text_encoder_lora_params),
        lr=args.learning_rate,
    )

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.max_train_steps,
    )

    unet, text_encoder, train_dataloader = accelerator.prepare(unet, text_encoder, train_dataloader)

    for epoch in range(args.num_train_epochs):
        unet.train()
        text_encoder.train()

        for step, batch in enumerate(train_dataloader):
            latents = vae.encode(batch["instance_images"].to(accelerator.device, dtype=torch.float16)).latent_dist.sample()
            latents = latents * 0.18215
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, 1000, (bsz,), device=latents.device).long()
            noisy_latents = noise * torch.sqrt(timesteps)
            encoder_hidden_states = text_encoder(batch["instance_prompt_ids"])[0]
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            if torch.isnan(loss):
                print("NaN loss detected!")
                break

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if step % args.save_steps == 0:
                if accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{epoch}-{step}")
                    os.makedirs(save_path, exist_ok=True)
                    save_lora_weight(unet, os.path.join(save_path, "unet.pt"))
                    save_lora_weight(text_encoder, os.path.join(save_path, "text_encoder.pt"))

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "final_checkpoint")
        os.makedirs(save_path, exist_ok=True)
        save_lora_weight(unet, os.path.join(save_path, "unet.pt"))
        save_lora_weight(text_encoder, os.path.join(save_path, "text_encoder.pt"))

    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)
