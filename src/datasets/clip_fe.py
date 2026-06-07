import torch
import os
import pandas as pd
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.datasets.dataset import EpicKitchensFramesDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

model.eval()


train_dataset = EpicKitchensFramesDataset(
    csv_file='./data/annotations/processed/train.csv', 
    frames_dir = './data/sampled/train', 
    tokenizer = None,
    mode='val'
)

val_dataset = EpicKitchensFramesDataset(
    csv_file='./data/annotations/processed/val.csv', 
    frames_dir = './data/sampled/val', 
    tokenizer = None,
    mode='val'
)

batch_size = 64 

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)



def extract_and_save(loader, output_file):
    dataset_features = {}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extracting to {output_file}"):
            narration_ids = batch['narration_id']
            video_tensor = batch['video'].to(device)
            
            text_inputs = processor(text=batch['raw_text'], return_tensors="pt", padding=True, truncation=True).to(device)
            text_features = model.get_text_features(**text_inputs)
            
            B, T, C, H, W = video_tensor.shape
            video_flat = video_tensor.view(B * T, C, H, W)
            
            image_features = model.get_image_features(pixel_values=video_flat) 
            
            image_features = image_features.pooler_output.view(B, T, -1) 
            video_features = image_features.mean(dim=1)    
        
            text_features = text_features.pooler_output.cpu()
            video_features = video_features.cpu()
            
            for i, n_id in enumerate(narration_ids):
                dataset_features[n_id] = {
                    'text': text_features[i],
                    'video': video_features[i]
                }
                
    torch.save(dataset_features, f'./data/features/{output_file}')


extract_and_save(train_loader, 'clip_features_train.pt')
extract_and_save(val_loader, 'clip_features_val.pt')