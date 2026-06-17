import torch
import pandas as pd
from transformers import AutoTokenizer
from src.models.adapter import AdapterModule

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer


"""text_encoder and text_proj derives from EgoVLP's text encoder"""

class InferenceModel(nn.Module):
    def __init__(self, text_encoder, text_proj, adapter):
        super().__init__()
        self.text_encoder = text_encoder
        self.text_proj = text_proj
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.adapter = adapter

    @torch.no_grad
    def forward(self, text):
        input = self.tokenizer(text, return_tensors="pt",
                       padding='max_length', max_length=32, truncation=True)
        input_ids = input['input_ids']
        attention_mask = input['attention_mask']
        
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        emb = self.text_proj(pooled)

        norm_embed = F.normalize(emb, dim=-1)
        final_embedding = self.adapter.forward(video_input=None, text_input=norm_embed)
        return final_embedding[1]


class InferencePipeline():
    def __init__(self, model_path, df_path, video_embeddings_path):
        self.model = torch.load(model_path, weights_only=False, map_location="cpu")
        self.model.eval()
        self.video_embeddings = torch.load(video_embeddings_path)
        self.df = pd.read_csv(df_path)

    def run(self, text):
        text_embedding = self.model(text)

        sim_matrix = torch.matmul(text_embedding, self.video_embeddings.T)

        top_10 = torch.topk(sim_matrix, 10, dim=1)
        
        videos = self.df.iloc[top_10.indices[0].tolist()]

        videos = videos[["video_id", "narration_id", "narration", "start_frame", "stop_frame"]]

        return videos