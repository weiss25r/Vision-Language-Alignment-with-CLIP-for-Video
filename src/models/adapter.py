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
            nn.Linear(in_dim, out_dim),
            # nn.LayerNorm(hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class Adapter(nn.Module):
    def __init__(self, video_mlp_config, text_mlp_config):
        super(Adapter, self).__init__()
        self.log_t = nn.Parameter(
            torch.zeros([])
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
        labels = torch.arange(batch_size, device=v_e.device)
        loss_t = F.cross_entropy(logits, labels)
        loss_v = F.cross_entropy(logits.T, labels)

        loss = (loss_t + loss_v) /2

        return loss, sim_matrix

class AdapterModule(LightningModule):
    def __init__(self, lr, weight_decay, adapter_config):
        super(AdapterModule, self).__init__()
        self.model = Adapter(adapter_config["video_mlp"], adapter_config["text_mlp"])
        self.val_video_embeddings = []
        self.val_text_embeddings = []
        self.save_hyperparameters()
        self.test_video_embeddings_seen = []
        self.test_text_embeddings_seen = []
        self.test_video_embeddings_zeroshot = []
        self.test_text_embeddings_zeroshot = []

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
        loss, _ = self.model.get_clip_loss(video_output, text_output)
        self.log('val/loss', loss,on_step=False, on_epoch=True)

        with torch.no_grad():
            v_e = F.normalize(video_output, dim=1).detach().cpu()
            t_e = F.normalize(text_output, dim=1).detach().cpu()

            self.val_video_embeddings.append(v_e)
            self.val_text_embeddings.append(t_e)
        return loss
    
    def on_validation_epoch_end(self):
        all_videos = torch.cat(self.val_video_embeddings)
        all_texts = torch.cat(self.val_text_embeddings)

        sim_matrix = torch.matmul(all_texts, all_videos.T)

        recalls = compute_recall(sim_matrix, len(all_videos), "val/")
        self.log_dict(recalls, on_step=False, on_epoch=True)

        self.val_video_embeddings.clear()
        self.val_text_embeddings.clear()
    
    def test_step(self, batch, batch_idx, dataloader_idx):
        text, video, = batch
        video_output, text_output = self.model(video, text)
        
        if dataloader_idx == 0:
            self.test_video_embeddings_seen.append(video_output.detach().cpu())
            self.test_text_embeddings_seen.append(text_output.detach().cpu())
        elif dataloader_idx == 1:
            self.test_video_embeddings_zeroshot.append(video_output.detach().cpu())
            self.test_text_embeddings_zeroshot.append(text_output.detach().cpu())
    
    def on_test_epoch_end(self):
        all_videos_seen = F.normalize(torch.cat(self.test_video_embeddings_seen), dim=1)
        all_texts_seen = F.normalize(torch.cat(self.test_text_embeddings_seen), dim = 1)

        all_videos_zeroshot = F.normalize(torch.cat(self.test_video_embeddings_zeroshot), dim=1)
        all_texts_zeroshot = F.normalize(torch.cat(self.test_text_embeddings_zeroshot), dim=1)

        sim_matrix_seen = torch.matmul(all_texts_seen, all_videos_seen.T)
        sim_matrix_zeroshot = torch.matmul(all_texts_zeroshot, all_videos_zeroshot.T)


        recalls_seen = compute_recall(sim_matrix_seen, len(all_videos_seen), "test-seen/")
        recalls_zeroshot = compute_recall(sim_matrix_zeroshot, len(all_videos_zeroshot), "test-zeroshot/")

        self.log_dict(recalls_seen, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(recalls_zeroshot, on_step=False, on_epoch=True, prog_bar=True)

        self.test_video_embeddings_seen.clear()
        self.test_text_embeddings_seen.clear()
        self.test_video_embeddings_zeroshot.clear()
        self.test_text_embeddings_zeroshot.clear()
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer