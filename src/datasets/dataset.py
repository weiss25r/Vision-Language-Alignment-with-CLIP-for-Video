import torch

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image
import numpy as np
import pandas as pd
import os
import cv2
import ast

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
        
        self.mode = mode

        if mode == 'val':
            self.transforms = v2.Compose([
                v2.ToImage(),
                v2.Resize(224, antialias=True),
                v2.CenterCrop(224),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=self.mean, std=self.std)
            ])
        elif mode == 'train':
            self.transforms = v2.Compose([
                v2.ToImage(),
                v2.Resize(256, antialias=True),
                v2.RandomCrop(224),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=self.mean, std=self.std)
            ])
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def __len__(self):
        
        return len(self.df)

    def __getitem__(self, idx):
        
        row = self.df.iloc[idx]
        narration_id = row['narration_id'] 
        video_id = row['video_id']
        
        #random for training, center for validation
        if self.mode == 'train':
            frame_indices = get_uniform_frame_indices(row['start_frame'], row['stop_frame'], num_frames=8, strategy='random')
        else:
            frame_indices = get_uniform_frame_indices(row['start_frame'], row['stop_frame'], num_frames=8, strategy='center')

        video_dir = os.path.join(self.frames_dir, video_id)
        
        frames = []
        for f_idx in frame_indices:
            frame_path = os.path.join(video_dir, f"{video_id}_frame_{f_idx:010d}.jpg")
            
            if not os.path.exists(frame_path):
                frame_path = os.path.join(video_dir, f"frame_{f_idx:010d}.jpg") 
            
            img_bgr = cv2.imread(frame_path)

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = self.transforms(img_rgb)
            frames.append(img_tensor)
        
        #video tensor: [T, C, H, W]
        video_tensor = torch.stack(frames, dim=0) 
        
        text = row['narration']

        if self.tokenizer is None:
            return {
                'narration_id': narration_id,
                'raw_text': text,
                'video': video_tensor
            }
        
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
            'text_attention_mask': text_inputs['attention_mask'],
            'verb': row['verb_class'],
            'noun': row['noun_class']
        }

class EpicKitchensFramesModule(LightningDataModule):
    def __init__(self, csv_dir, frames_dir, tokenizer, batch_size, num_workers):
        super().__init__()
        self.csv_dir = csv_dir
        self.frames_dir = frames_dir
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = EpicKitchensFramesDataset(os.path.join(self.csv_dir, 'train.csv'), os.path.join(self.frames_dir, 'train'), self.tokenizer, mode='train')
            self.val_dataset = EpicKitchensFramesDataset(os.path.join(self.csv_dir, 'val.csv'), os.path.join(self.frames_dir, 'val'), self.tokenizer, mode='val')
        if stage == 'test':
            self.test_seen_dataset = EpicKitchensFramesDataset(os.path.join(self.csv_dir, 'test_seen.csv'), os.path.join(self.frames_dir, 'test_seen'), self.tokenizer, mode='val')
            self.test_zeroshot_dataset = EpicKitchensFramesDataset(os.path.join(self.csv_dir, 'test_zeroshot.csv'), os.path.join(self.frames_dir, 'test_zeroshot'), self.tokenizer, mode='val')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else 0
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else 0
        )

    def test_dataloader(self):
        dl_seen = DataLoader(self.test_seen_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        dl_zeroshot = DataLoader(self.test_zeroshot_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return [dl_seen, dl_zeroshot]

# In src/datasets/dataset.py

class EpicKitchensFeatureDataset(Dataset):
    def __init__(self, features_path, csv_path=None):
        with open(features_path, 'rb') as f:
            self.features = torch.load(f)
            self.keys = list(self.features.keys())

        if csv_path is not None:
            self.df = pd.read_csv(csv_path, index_col="narration_id")
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        tensor = self.features[self.keys[idx]]


        verb = self.df.loc[self.keys[idx]]['verb_class']
        noun = self.df.loc[self.keys[idx]]['noun_class']

        return (tensor['text'], tensor['video'], verb, noun)
    
class EpicKitchensFeatureModule(LightningDataModule):
    def __init__(self, features_dir, csv_dir, batch_size, num_workers):
        super().__init__()
        self.features_dir = features_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.csv_dir = csv_dir
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = EpicKitchensFeatureDataset(os.path.join(self.features_dir, 'features_train.pt'), os.path.join(self.csv_dir, 'train.csv'))
            self.val_dataset = EpicKitchensFeatureDataset(os.path.join(self.features_dir, 'features_val.pt'), os.path.join(self.csv_dir, 'val.csv'))        
        if stage == 'test':
            self.test_seen_dataset = EpicKitchensFeatureDataset(os.path.join(self.features_dir, 'features_test_seen.pt'), os.path.join(self.csv_dir, 'test_seen.csv'))
            self.test_zeroshot_dataset = EpicKitchensFeatureDataset(os.path.join(self.features_dir, 'features_test_zeroshot.pt'), os.path.join(self.csv_dir, 'test_zeroshot.csv'))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True
        )

    def test_dataloader(self):
        dl_seen = DataLoader(self.test_seen_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        dl_zeroshot = DataLoader(self.test_zeroshot_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return [dl_seen, dl_zeroshot]