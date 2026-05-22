import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch.optim import AdamW

from ..evaluation.metrics import compute_recall
from lightning.pytorch.loggers import WandbLogger

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

import numpy as np

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super(MLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class Adapter(nn.Module):
    def __init__(self, video_mlp_config, text_mlp_config):
        super(Adapter, self).__init__()
        self.log_t = nn.Parameter(
            torch.ones([]) * np.log(1/ 0.07)
        )
        
        self.video_mlp = MLP(**video_mlp_config)
        self.text_mlp = MLP(**text_mlp_config)

    def forward(self, video_input, text_input):
        video_output = self.video_mlp(video_input)
        text_output = self.text_mlp(text_input)
        return video_output, text_output
    
    def get_clip_loss(self, video, text):
        v_e = F.normalize(video, dim=1)
        t_e = F.normalize(text, dim=1)
        sim_matrix = torch.matmul(t_e, v_e.T)

        temperature = self.log_t.exp().clamp(max=100.0)

        logits = sim_matrix * temperature
        batch_size = v_e.size(0)
        labels = torch.arange(batch_size, device="mps")
        loss_t = F.cross_entropy(logits, labels)
        loss_v = F.cross_entropy(logits.T, labels)

        loss = (loss_t + loss_v) /2

        return loss, sim_matrix

class AdapterModule(LightningModule):
    def __init__(self, lr, weight_decay, adapter_config):
        super(AdapterModule, self).__init__()
        self.model = Adapter(adapter_config["video_mlp"], adapter_config["text_mlp"])
        self.save_hyperparameters()

    def forward(self, video_input, text_input):
        video_output, text_output = self.model(video_input, text_input)
        return video_output, text_output
    
    def training_step(self, batch, batch_idx):
        video, text = batch
        video_output, text_output = self.model(video, text)
        loss, _ = self.model.get_clip_loss(video_output, text_output)
        self.log('train/loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        video, text = batch
        video_output, text_output = self.model(video, text)
        loss, sim_matrix = self.model.get_clip_loss(video_output, text_output)
        self.log('val/loss', loss)
        recalls = compute_recall(sim_matrix, len(video_output), "val/")

        self.log_dict(recalls)
        return loss
    
    def test_step(self, *args, **kwargs):
        #define seen and zero-shot
        pass
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer