import torch
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.datasets.dataset import EpicKitchensFramesDataset
from src.models.encoders import TextEncoder, VideoEncoder
from transformers import DistilBertModel, TimesformerModel
from transformers import AutoTokenizer
from tqdm import tqdm

distilbert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

train_dataset = EpicKitchensFramesDataset(
    csv_file='../data/annotations/processed/train.csv', 
    frames_dir = '../data/sampled/train', 
    tokenizer = distilbert_tokenizer,
    mode='val'
)

val_seen_dataset = EpicKitchensFramesDataset(
    csv_file='../data/annotations/processed/val_seen.csv', 
    frames_dir = '../data/sampled/val_seen', 
    tokenizer = distilbert_tokenizer,
    mode='val'
)

val_zeroshot = EpicKitchensFramesDataset(
    csv_file='../data/annotations/processed/val_zeroshot.csv', 
    frames_dir = '../data/sampled/val_zeroshot', 
    tokenizer = distilbert_tokenizer,
    mode='val'
)

batch_size = 128

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False
)

val_seen_loader = torch.utils.data.DataLoader(
    val_seen_dataset,
    batch_size=batch_size,
    shuffle=False
)

val_zeroshot_loader = torch.utils.data.DataLoader(
    val_zeroshot,
    batch_size=batch_size,
    shuffle=False
)

val_zeroshot_loader = torch.utils.data.DataLoader(
    val_zeroshot,
    batch_size=batch_size,
    shuffle=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

text_model = DistilBertModel.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
text_encoder = TextEncoder(text_model).to(device)

video_model = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k600")
video_encoder = VideoEncoder(video_model).to(device)

text_encoder.eval().to(device)
video_encoder.eval().to(device)

with torch.no_grad():
    
    text_encoder.eval().to(device)
    video_encoder.eval().to(device)

    files = ['features_train.pt', 'features_val_seen.pt', 'features_val_zeroshot.pt']
    k = 0
    for loader in [train_loader, val_seen_loader, val_zeroshot_loader]:
        
        dataset_features = {}
        for batch in tqdm(loader, desc=f"Extracting Features for loader {k}"):
            
            narration_ids = batch['narration_id']
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            video_tensor = batch['video'].to(device)
            
            text_output = text_encoder(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask
            )
           
            text_embeddings = text_output.last_hidden_state[:, 0, :].cpu()
            
            video_output = video_encoder(video_tensor)
            
            video_embeddings = video_output.last_hidden_state[:, 0, :].cpu()
            
            for i, n_id in enumerate(narration_ids):
                dataset_features[n_id] = {
                    'text': text_embeddings[i],
                    'video': video_embeddings[i]
                }
        
        torch.save(dataset_features, '../data/features/' + files[k])
        k = k +1

print("Process completed successfully")