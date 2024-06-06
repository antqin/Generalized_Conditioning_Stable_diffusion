from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from read_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

#torch.cuda.empty_cache()

# Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = 1
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = True


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=12, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
#print("Made it past the image logger")
trainer = pl.Trainer(gpus=2, accumulate_grad_batches=4, strategy="ddp", precision=16, callbacks=[logger])


# Train!
trainer.fit(model, dataloader)
