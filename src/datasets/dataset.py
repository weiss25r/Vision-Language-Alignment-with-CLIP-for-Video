import torch

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image
import numpy as np
import pandas as pd
import os

from torch.utils.data import DataLoader

from lightning import LightningDataModule

from ..utils.sampling_utils import get_uniform_frame_indices

class EpicKitchensFramesDataset(Dataset):
    def __init__(self, csv_file, frames_dir, tokenizer, mode="val"):
        self.df = pd.read_csv(csv_file)
        self.frames_dir = frames_dir
        self.tokenizer = tokenizer

        # Normalization using mean and std of ImageNet
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        if mode == 'val':
            # Transforms for inference mode
            self.transforms = v2.Compose([
                v2.ToImage(),
                v2.Resize(224, antialias=True),
                v2.CenterCrop(224),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=self.mean, std=self.std)
            ])

        #TODO: Add transforms for training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        row = self.df.iloc[idx]
        narration_id = row['narration_id'] 
        video_id = row['video_id']
        
        frame_indices = get_uniform_frame_indices(row['start_frame'], row['stop_frame'], strategy='center')
        video_dir = os.path.join(self.frames_dir, video_id)
        
        frames = []
        for f_idx in frame_indices:
            frame_path = os.path.join(video_dir, f"{video_id}_frame_{f_idx:010d}.jpg")
            
            if not os.path.exists(frame_path):
                frame_path = os.path.join(video_dir, f"frame_{f_idx:010d}.jpg") 
                
            img = Image.open(frame_path).convert('RGB')
            img_tensor = self.transforms(img) # [C, H, W]
            frames.append(img_tensor)
        
        #video tensor: [T, C, H, W]
        video_tensor = torch.stack(frames, dim=0) 
        
        text = row['narration']
        text_inputs = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=32,
            return_tensors="pt"
        )
        
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}

        return {
            'narration_id': narration_id,
            'video': video_tensor,
            'text_input_ids': text_inputs['input_ids'],
            'text_attention_mask': text_inputs['attention_mask']
        }
        

class EpicKitchensFeatureDataset(Dataset):
    def __init__(self, features_path):
        with open(features_path, 'rb') as f:
            self.features = torch.load(f)
            self.keys = list(self.features.keys())
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        tensor = self.features[self.keys[idx]]
        return (tensor['text'], tensor['video'])
    
class EpicKitchensFeatureModule(LightningDataModule):
    def __init__(self, features_dir, batch_size, num_workers):
        super().__init__()
        self.features_dir = features_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = EpicKitchensFeatureDataset(os.path.join(self.features_dir, 'features_train.pt'))
            self.val_dataset = EpicKitchensFeatureDataset(os.path.join(self.features_dir, 'features_val.pt'))        
        if stage == 'test':
            self.test_seen_dataset = EpicKitchensFeatureDataset(os.path.join(self.features_dir, 'features_test_seen.pt'))
            self.test_zeroshot_dataset = EpicKitchensFeatureDataset(os.path.join(self.features_dir, 'features_test_zeroshot.pt'))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        dl_seen = DataLoader(self.test_seen_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        dl_zeroshot = DataLoader(self.test_zeroshot_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return [dl_seen, dl_zeroshot]