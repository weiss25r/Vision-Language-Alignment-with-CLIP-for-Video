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
from ..evaluation.metrics import compute_recall, compute_multi_instance_recall

from torch.optim import AdamW

from transformers import get_cosine_schedule_with_warmup

class VideoCLIP(nn.Module):
    def __init__(self, text_encoder, video_encoder, video_mlp_config, text_mlp_config):
        super(VideoCLIP, self).__init__()

        self.log_t = nn.Parameter(
            torch.ones([]) * np.log(1/ 0.07)
        )

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


        video_output = self.video_encoder(video)
        video_embedding = video_output.last_hidden_state[:, 0, :]

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
        labels = torch.arange(batch_size, device=v_e.device)
        loss_t = F.cross_entropy(logits, labels)
        loss_v = F.cross_entropy(logits.T, labels)

        loss = (loss_t + loss_v) /2

        return loss, sim_matrix
    
class VideoCLIPModule(LightningModule):
    def __init__(self, lr, weight_decay, adapter_config):
        super(VideoCLIPModule, self).__init__()

        text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        text_model.train()
        text_encoder = TextEncoder(text_model)

        video_model = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k600")
        video_encoder = VideoEncoder(video_model)
        video_encoder.eval()

        for param in video_encoder.parameters():
            param.requires_grad = False

        layers_to_unfreeze = ["layer.11."]
        final_layernorm = {"layernorm.weight", "layernorm.bias"}
        
        for name, param in video_encoder.named_parameters():
            if any(target in name for target in layers_to_unfreeze) or name in final_layernorm:
                param.requires_grad = True

        self.model = VideoCLIP(
            text_encoder,
            video_encoder,
            adapter_config["video_mlp"],
            adapter_config["text_mlp"]
        )

        self.verb_classes_seen = []
        self.noun_classes_seen = []
        self.verb_classes_zeroshot = []
        self.noun_classes_zeroshot = []

        self.test_video_embeddings_seen = []
        self.test_text_embeddings_seen = []
        self.test_video_embeddings_zeroshot = []
        self.test_text_embeddings_zeroshot = []

        self.save_hyperparameters()

    def forward(self, video, text_input_ids, text_attention_mask):
        video_output, text_output = self.model(video, text_input_ids, text_attention_mask)
        return video_output, text_output

    def on_train_epoch_start(self):
        self.model.video_encoder.eval()
        for name, module in self.model.video_encoder.named_modules():
            if "layer.11" in name or name == "layernorm":
                module.train()

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
        verb = batch['verb']
        noun = batch['noun']

        video_output, text_output = self.model(video_tensor, text_input_ids, text_attention_mask)
        loss, _ = self.model.get_clip_loss(video_output, text_output)
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
        text_input_ids = batch['text_input_ids']
        text_attention_mask = batch['text_attention_mask']
        video_tensor = batch['video']
        verb = batch['verb']
        noun = batch['noun']

        video_output, text_output = self.model(video_tensor, text_input_ids, text_attention_mask)

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
            self.verb_classes_zeroshot.append(verb.cpu())
            self.noun_classes_zeroshot.append(noun.cpu())

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
            sim_matrix_seen, all_verbs_seen, all_nouns_seen, "test-seen/"
        )
        recalls_zeroshot = compute_multi_instance_recall(
            sim_matrix_zeroshot, all_verbs_zeroshot, all_nouns_zeroshot, "test-zeroshot/"
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
        fast_params = (
            list(self.model.text_encoder.parameters()) +
            list(self.model.video_mlp.parameters()) +
            list(self.model.text_mlp.parameters())
        )
        slow_params = [p for p in self.model.video_encoder.parameters() if p.requires_grad]

        optimizer = AdamW(
            [
                {'params': fast_params, 'lr': self.hparams.lr},
                {'params': slow_params, 'lr': self.hparams.lr * 0.1}
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