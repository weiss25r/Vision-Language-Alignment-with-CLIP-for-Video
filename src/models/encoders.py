import torch
import torch.nn as nn
import sys
import os
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, encoder_model):
        super(TextEncoder, self).__init__()
        self.model = encoder_model

    def forward(self, text_input_ids, text_attention_mask):
        output = self.model(
            input_ids=text_input_ids, 
            attention_mask=text_attention_mask
        )
        return output
    
class VideoEncoder(nn.Module):
    def __init__(self, encoder_model):
        super(VideoEncoder, self).__init__()
        self.model = encoder_model

    def forward(self, video_input):
        output = self.model(pixel_values=video_input)
        return output

class EgoVLPVideoEncoder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, video: torch.Tensor):
        video_embedding = self.model.compute_video(video)
        return video_embedding

class EgoVLPTextEncoder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, text_input_ids: torch.Tensor, text_attention_mask: torch.Tensor):
        text_embedding = self.model.compute_text({
            'input_ids': text_input_ids,
            'attention_mask': text_attention_mask
        })
        return text_embedding
    
#TODO: refactor
