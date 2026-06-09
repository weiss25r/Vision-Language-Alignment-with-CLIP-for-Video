import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch.optim import AdamW

from ..evaluation.metrics import compute_recall, compute_multi_instance_recall
from lightning.pytorch.loggers import WandbLogger

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

import numpy as np

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super(MLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)

class Adapter(nn.Module):
    def __init__(self, video_mlp_config, text_mlp_config, loss="egonce"):
        super(Adapter, self).__init__()
        self.log_t = nn.Parameter(
            torch.zeros([])
        )

        self.video_mlp = MLP(**video_mlp_config)
        self.text_mlp = MLP(**text_mlp_config)
        self.loss = loss

    def forward(self, video_input, text_input):
        video_output = self.video_mlp(video_input)
        text_output = self.text_mlp(text_input)
        return video_output, text_output
    

    def get_loss(self, **kwargs):
        if self.loss == "egonce":
            return self.egonce_loss(**kwargs)
        elif self.loss == "clip":
            return self.get_clip_loss(**kwargs)
        else:
            raise ValueError(f"Invalid loss: {self.loss}")


    def get_clip_loss(self, video_features, text_features, **kwargs):
        v_e = F.normalize(video_features, dim=1)
        t_e = F.normalize(text_features, dim=1)
        sim_matrix = torch.matmul(t_e, v_e.T)

        temperature = self.log_t.exp().clamp(max=100.0)

        logits = sim_matrix * temperature
        batch_size = v_e.size(0)
        labels = torch.arange(batch_size, device=v_e.device)
        loss_t = F.cross_entropy(logits, labels)
        loss_v = F.cross_entropy(logits.T, labels)

        loss = (loss_t + loss_v) /2

        return loss
    

    def egonce_loss(self, video_features, text_features, verb_classes, noun_classes, temperature=0.05, **kwargs):
        video_features = F.normalize(video_features, dim=1)
        text_features = F.normalize(text_features, dim=1)
        
        device = video_features.device

        sim_matrix_v2t = torch.matmul(video_features, text_features.T) / temperature
        sim_matrix_t2v = torch.matmul(text_features, video_features.T) / temperature
        
        verb_mask = (verb_classes.unsqueeze(0) == verb_classes.unsqueeze(1))
        noun_mask = (noun_classes.unsqueeze(0) == noun_classes.unsqueeze(1))
        
        positive_mask = (verb_mask & noun_mask).float().to(device)
        
        exp_sim_v2t = torch.exp(sim_matrix_v2t)
        numerator_v2t = (exp_sim_v2t * positive_mask).sum(dim=1)
        
        denominator_v2t = exp_sim_v2t.sum(dim=1)
        
        loss_v2t = -torch.log(numerator_v2t / denominator_v2t + 1e-9).mean()
        
        exp_sim_t2v = torch.exp(sim_matrix_t2v)
        numerator_t2v = (exp_sim_t2v * positive_mask).sum(dim=1)
        denominator_t2v = exp_sim_t2v.sum(dim=1)
        loss_t2v = -torch.log(numerator_t2v / denominator_t2v + 1e-9).mean()
        
        return (loss_v2t + loss_t2v) / 2

class AdapterModule(LightningModule):
    def __init__(self, lr, weight_decay, adapter_config, loss="egonce"):
        super(AdapterModule, self).__init__()
        self.model = Adapter(adapter_config["video_mlp"], adapter_config["text_mlp"], loss)

        self.save_hyperparameters()

        self.verb_classes_seen = []
        self.noun_classes_seen = []

        self.verb_classes_zeroshot = []
        self.noun_classes_zeroshot = []

        self.test_video_embeddings_seen = []
        self.test_text_embeddings_seen = []
        self.test_video_embeddings_zeroshot = []
        self.test_text_embeddings_zeroshot = []

    def forward(self, video_input, text_input):
        video_output, text_output = self.model(video_input, text_input)
        return video_output, text_output
    
    def training_step(self, batch, batch_idx):
        text, video, verb, noun = batch
        video_output, text_output = self.model(video, text)

        loss = self.model.get_loss(
            video_features=video_output,
            text_features=text_output,
            verb_classes=verb,
            noun_classes=noun,
            temperature=0.05
        )
        self.log('train/loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        text, video, verb, noun = batch
        video_output, text_output = self.model(video, text)

        loss = self.model.get_loss(
            video_features=video_output,
            text_features=text_output,
            verb_classes=verb,
            noun_classes=noun,
            temperature=0.05
        )
        self.log('val/loss', loss, on_step=False, on_epoch=True)

        with torch.no_grad():
            v_e = F.normalize(video_output, dim=1).detach().cpu()
            t_e = F.normalize(text_output, dim=1).detach().cpu()

            self.test_video_embeddings_seen.append(v_e)
            self.test_text_embeddings_seen.append(t_e)

            self.verb_classes_seen.append(verb.cpu())
            self.noun_classes_seen.append(noun.cpu())
        return loss
    
    def on_validation_epoch_end(self):
        all_videos = torch.cat(self.test_video_embeddings_seen)
        all_texts = torch.cat(self.test_text_embeddings_seen)
        all_verbs = torch.cat(self.verb_classes_seen)
        all_nouns = torch.cat(self.noun_classes_seen)

        sim_matrix = torch.matmul(all_texts, all_videos.T)

        recalls = compute_multi_instance_recall(sim_matrix, all_verbs, all_nouns, "val/")
        self.log_dict(recalls, on_step=False, on_epoch=True)

        self.test_video_embeddings_seen.clear()
        self.test_text_embeddings_seen.clear()

        self.verb_classes_seen.clear()
        self.noun_classes_seen.clear()
    
    def test_step(self, batch, batch_idx, dataloader_idx):
        text, video, verb, noun = batch
        video_output, text_output = self.model(video, text)
        
        v_e = F.normalize(video_output, dim=1).detach().cpu()
        t_e = F.normalize(text_output, dim=1).detach().cpu()

        if dataloader_idx == 0:
            self.test_video_embeddings_seen.append(v_e)
            self.test_text_embeddings_seen.append(t_e)

            self.verb_classes_seen.append(verb.cpu())
            self.noun_classes_seen.append(noun.cpu())
        elif dataloader_idx == 1:
            self.test_video_embeddings_zeroshot.append(v_e)
            self.test_text_embeddings_zeroshot.append(t_e)

            self.verb_classes_zeroshot.append(verb)
            self.noun_classes_zeroshot.append(noun)
    
    def on_test_epoch_end(self):
        all_videos_seen = torch.cat(self.test_video_embeddings_seen)
        all_texts_seen = torch.cat(self.test_text_embeddings_seen)

        all_videos_zeroshot = torch.cat(self.test_video_embeddings_zeroshot)
        all_texts_zeroshot = torch.cat(self.test_text_embeddings_zeroshot)

        all_verbs_seen = torch.cat(self.verb_classes_seen)
        all_nouns_seen = torch.cat(self.noun_classes_seen)

        all_verbs_zeroshot = torch.cat(self.verb_classes_zeroshot)
        all_nouns_zeroshot = torch.cat(self.noun_classes_zeroshot)

        sim_matrix_seen = torch.matmul(all_texts_seen, all_videos_seen.T)
        sim_matrix_zeroshot = torch.matmul(all_texts_zeroshot, all_videos_zeroshot.T)


        recalls_seen = compute_multi_instance_recall(
            sim_matrix_seen,
            all_verbs_seen,
            all_nouns_seen,
            "test-seen/"
        )

        recalls_zeroshot = compute_multi_instance_recall(
            sim_matrix_zeroshot,
            all_verbs_zeroshot,
            all_nouns_zeroshot,
            "test-zeroshot/"
        )

        self.log_dict(recalls_seen, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(recalls_zeroshot, on_step=False, on_epoch=True, prog_bar=True)

        self.test_video_embeddings_seen.clear()
        self.test_text_embeddings_seen.clear()
        self.test_video_embeddings_zeroshot.clear()
        self.test_text_embeddings_zeroshot.clear()

        self.verb_classes_seen.clear()
        self.noun_classes_seen.clear()
        self.verb_classes_zeroshot.clear()
        self.noun_classes_zeroshot.clear()
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer