import torch

from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

from src.models.adapter import MLP

import numpy as np
from src.models.encoders import VideoEncoder, TextEncoder
from transformers import DistilBertModel, TimesformerModel

from lightning.pytorch import LightningModule
from ..evaluation.metrics import compute_recall

from torch.optim import AdamW

from transformers import get_cosine_schedule_with_warmup

class VideoCLIP(nn.Module):
    def __init__(self, text_encoder, video_encoder, video_mlp_config, text_mlp_config):
        super(VideoCLIP, self).__init__()
        self.log_t = nn.Parameter(
            torch.ones([]) * np.log(1/ 0.07)
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        self.video_encoder = video_encoder
        self.text_encoder = text_encoder
        self.video_mlp = MLP(**video_mlp_config)
        self.text_mlp = MLP(**text_mlp_config)

    def forward(self, video, text_input_ids, text_attention_mask):
        text_output = self.text_encoder(
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask
        )

        input_mask_expanded = text_attention_mask.unsqueeze(-1).expand(text_output.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(text_output.last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)
        text_embedding = (sum_embeddings / sum_mask)


        with torch.no_grad():
            video_output = self.video_encoder(video)
        video_embedding = video_output.last_hidden_state.mean(dim=1) 

        video_output = self.video_mlp(video_embedding)
        text_output = self.text_mlp(text_embedding)
        return video_output, text_output
    
    def get_clip_loss(self, video, text):
        v_e = F.normalize(video, dim=1)
        t_e = F.normalize(text, dim=1)
        sim_matrix = torch.matmul(t_e, v_e.T)

        temperature = self.log_t.exp().clamp(max=100.0)

        logits = sim_matrix * temperature
        batch_size = v_e.size(0)
        labels = torch.arange(batch_size, device=self.device)
        loss_t = F.cross_entropy(logits, labels)
        loss_v = F.cross_entropy(logits.T, labels)

        loss = (loss_t + loss_v) /2

        return loss, sim_matrix
    
class VideoCLIPModule(LightningModule):
    def __init__(self, lr, weight_decay, adapter_config):
        super(VideoCLIPModule, self).__init__()

        text_model = DistilBertModel.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        text_model.train()

        text_encoder = TextEncoder(text_model)

        video_model = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k600")
        video_encoder = VideoEncoder(video_model)
        video_encoder.eval()

        for param in video_encoder.parameters():
            param.requires_grad = False
        
        layers_to_unfreeze = ["layer.11.", "layernorm"]
        for name, param in video_encoder.named_parameters():
            if any(target in name for target in layers_to_unfreeze):
                param.requires_grad = True
        
        self.model = VideoCLIP(
            text_encoder,
            video_encoder,
            adapter_config["video_mlp"], 
            adapter_config["text_mlp"])
        
        self.val_video_embeddings = []
        self.val_text_embeddings = []

        self.save_hyperparameters()

    def forward(self, video, text_input_ids, text_attention_mask):
        video_output, text_output = self.model(video, text_input_ids, text_attention_mask)
        return video_output, text_output
    
    def training_step(self, batch, batch_idx):
        text_input_ids = batch['text_input_ids']
        text_attention_mask = batch['text_attention_mask']
        video_tensor = batch['video']

        video_output, text_output = self.model(video_tensor, text_input_ids, text_attention_mask)
        loss, _ = self.model.get_clip_loss(video_output, text_output)
        self.log('train/loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):

        text_input_ids = batch['text_input_ids']
        text_attention_mask = batch['text_attention_mask']
        video_tensor = batch['video']

        video_output, text_output = self.model(video_tensor, text_input_ids, text_attention_mask)
    
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
    def test_step(self, *args, **kwargs):
        #define seen and zero-shot
        pass
    
    def configure_optimizers(self):

        fast_params = (
            list(self.model.text_encoder.parameters()) +
            list(self.model.video_mlp.parameters()) +
            list(self.model.text_mlp.parameters())
        )

        slow_params = [p for p in self.model.video_encoder.parameters() if p.requires_grad]
        
        lr_text = self.hparams.lr
        lr_video = self.hparams.lr * 0.1

        optimizer = AdamW(
            [
                {'params': fast_params, 'lr': lr_text},
                {'params': slow_params, 'lr': lr_video}
            ],
            weight_decay=self.hparams.weight_decay
        )
        total_steps = self.trainer.estimated_stepping_batches
        
        warmup_steps = int(total_steps * 0.1)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }