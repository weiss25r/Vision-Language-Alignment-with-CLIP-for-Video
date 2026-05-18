#Simple MLP
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch.optim import AdamW

from lightning.pytorch.loggers import WandbLogger

from torch.optim.lr_scheduler import CosineAnnealingLR

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class VideoCLIP(nn.Module):
    def __init__(self):
        super(VideoCLIP, self).__init__()
        
        self.video_mlp = MLP(in_dim=768, hidden_dim=768, out_dim=512)
        self.text_mlp = MLP(in_dim=768, hidden_dim=768, out_dim=512)

    def forward(self, video_input, text_input):
        video_output = self.video_mlp(video_input)
        text_output = self.text_mlp(text_input)
        return video_output, text_output

class VideoCLIPModule(LightningModule):
    def __init__(self, temperature, lr, weight_decay, max_epochs, eta_min):
        super(VideoCLIPModule, self).__init__()
        self.model = VideoCLIP()

        self.save_hyperparameters()

    def forward(self, video_input, text_input):
        video_output, text_output = self.model(video_input, text_input)
        return video_output, text_output
    
    def training_step(self, batch, batch_idx):
        video, text = batch
        video_output, text_output = self.model(video, text)
        loss = self.get_clip_loss(video_output, text_output)
        self.log('train_loss', loss)
        print("TRAINING STEP: ", loss.item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        video, text = batch
        video_output, text_output = self.model(video, text)
        loss = self.get_clip_loss(video_output, text_output)
        self.log('val_loss', loss)
        return loss
    
    def get_clip_loss(self, video, text):
        v_e = F.normalize(video, dim=1)
        t_e = F.normalize(text, dim=1)
        logits = torch.matmul(v_e, t_e.T)
        logits = logits / self.hparams.temperature
        batch_size = v_e.size(0)
        print(batch_size)
        labels = torch.arange(batch_size, device=self.device)
        print(labels)
        loss_v = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)

        loss = (loss_t + loss_v) /2

        print(loss)
        return loss
    
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
    
    def lr_schedulers(self):
        return CosineAnnealingLR(self.optimizers(), T_max=self.hparams.max_epochs, eta_min=self.hparams.eta_min)