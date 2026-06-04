"""
Offline feature extraction with EgoVLP.
"""

import torch
import os
import sys
from tqdm import tqdm
from transformers import AutoTokenizer

CHECKPOINT_PATH = os.path.abspath("./EgoVLP-main/epic_mir_plus.pth") #fine-tuned on EPIC KITCHENS 100
OUTPUT_DIR      = "./data/features/egovlp_plus"
BATCH_SIZE      = 16  
NUM_WORKERS     = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.datasets.dataset import EpicKitchensFramesDataset
from src.models.encoders import EgoVLPVideoEncoder, EgoVLPTextEncoder
from src.utils.egovlp_utils import load_egovlp

SPLITS = {
    "features_train.pt":         ("./data/annotations/processed/train.csv",         "./data/sampled/train"),
    "features_val.pt":           ("./data/annotations/processed/val.csv",           "./data/sampled/val"),
    "features_test_seen.pt":     ("./data/annotations/processed/test_seen.csv",     "./data/sampled/test_seen"),
    "features_test_zeroshot.pt": ("./data/annotations/processed/test_zeroshot.csv", "./data/sampled/test_zeroshot"),
}



def extract_features(loader, video_encoder, text_encoder, device, filename):
    dataset_features = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc=filename):
            narration_ids       = batch['narration_id']
            text_input_ids      = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            video_tensor        = batch['video'].to(device)

            video_emb = video_encoder(video_tensor).cpu()
            text_emb  = text_encoder(text_input_ids, text_attention_mask).cpu()

            for i, n_id in enumerate(narration_ids):
                dataset_features[n_id] = {
                    'video': video_emb[i],
                    'text':  text_emb[i]
                }
    return dataset_features


if __name__ == '__main__':   
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    egovlp_model  = load_egovlp(CHECKPOINT_PATH, device)
    video_encoder = EgoVLPVideoEncoder(egovlp_model).to(device)
    text_encoder  = EgoVLPTextEncoder(egovlp_model).to(device)
    tokenizer     = AutoTokenizer.from_pretrained("distilbert-base-uncased")


    dummy_text = tokenizer("test", return_tensors="pt", padding='max_length', 
                           max_length=32, truncation=True)
    dummy_video = torch.zeros(1, 16, 3, 224, 224).to(device)
    
    with torch.no_grad():
        v_emb = video_encoder(dummy_video)
        t_emb = text_encoder(
            dummy_text['input_ids'].to(device), 
            dummy_text['attention_mask'].to(device)
        )
    
    print(f"Video embedding dim: {v_emb.shape}")
    print(f"Text embedding dim:  {t_emb.shape}")
    

    for filename, (csv_path, frames_dir) in SPLITS.items():
        output_path = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(output_path):
            print(f"Skipping {filename}: file already exists.")
            continue

        print(f"\nFeature extraction : {filename}")

        dataset = EpicKitchensFramesDataset(
            csv_file=csv_path,
            frames_dir=frames_dir,
            tokenizer=tokenizer,
            mode='val'
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS, 
            pin_memory=(device.type == 'cuda')
        )

        dataset_features = extract_features(
            loader, video_encoder, text_encoder, device, filename
        )

        torch.save(dataset_features, output_path)
        print(f"Saved: {output_path} ({len(dataset_features)} samples)")

    print("\nProcess completed.")