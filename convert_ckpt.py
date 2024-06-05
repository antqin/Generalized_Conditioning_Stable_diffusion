import torch
from safetensors.torch import load_file as load_safetensors

# Load the .safetensors weights
safetensors_path = "./finetune-lora-dreambooth-output/pytorch_lora_weights.safetensors"
ckpt_path = "./finetune-lora-dreambooth-output/dreambooth-model.ckpt"

state_dict = load_safetensors(safetensors_path)

# Save the state_dict as .ckpt
torch.save(state_dict, ckpt_path)
print(f"Model weights saved to {ckpt_path}")
