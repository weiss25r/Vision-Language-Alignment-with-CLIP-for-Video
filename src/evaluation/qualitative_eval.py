import argparse
import os
import sys

import torch
import torch.nn.functional as F
import pandas as pd
import yaml
from tqdm import tqdm
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.adapter import AdapterModule
from src.datasets.dataset import EpicKitchensFeatureDataset


class EpicKitchensFeatureDatasetWithMeta(EpicKitchensFeatureDataset):
    def __getitem__(self, idx):
        text_feat, video_feat, verb, noun = super().__getitem__(idx)
        narration_id = self.keys[idx]
        narration_text = self.df.loc[narration_id]['narration']
        return text_feat, video_feat, verb, noun, narration_id, narration_text


@torch.no_grad()
def collect_embeddings(model: AdapterModule, loader: DataLoader, device: torch.device):
    model.eval()
    model.to(device)

    all_video, all_text, meta = [], [], []

    for batch in tqdm(loader, desc="Collecting embeddings"):
        text_feat, video_feat, verb, noun, narration_ids, narration_texts = batch

        text_feat = text_feat.to(device)
        video_feat = video_feat.to(device)

        video_out, text_out = model(video_feat, text_feat)

        all_video.append(F.normalize(video_out, dim=1).cpu())
        all_text.append(F.normalize(text_out, dim=1).cpu())

        for i in range(len(narration_ids)):
            meta.append({
                'narration':  narration_texts[i],
                'verb_class': verb[i].item(),
                'noun_class': noun[i].item(),
            })

    return torch.cat(all_video), torch.cat(all_text), meta


def print_top10_r1_errors(all_video, all_text, meta, split_name):
    N = all_text.shape[0]
    
    sim_matrix = torch.matmul(all_text, all_video.T)
    top1_indices = sim_matrix.argmax(dim=1)

    records = []
    
    for i in range(N):
        top1_idx = top1_indices[i].item()
        
        if (meta[i]['verb_class'] != meta[top1_idx]['verb_class']) or \
           (meta[i]['noun_class'] != meta[top1_idx]['noun_class']):
            
            records.append({
                'Query (GT)': meta[i]['narration'],
                'Predetta (top-1)': meta[top1_idx]['narration'],
                'Sim coseno': round(sim_matrix[i, top1_idx].item(), 4)
            })
            
            if len(records) >= 10:
                break

    print(f"\n{'='*60}")
    print(f"--- Example R@1 errors (max 10) | Split: {split_name.upper()} ---")
    print(f"{'='*60}")
    
    if not records:
        print("No errors!")
    else:
        df = pd.DataFrame(records)
        print(df.to_string(index=False))
    print("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,  help='Path al config YAML')
    parser.add_argument('--ckpt',   type=str, required=True,  help='Path al checkpoint .ckpt')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    seed_everything(config.get('seed', 42))

    train_config = config['train_config']
    features_dir = train_config['feature_dir']
    csv_dir      = train_config['csv_dir']

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = AdapterModule.load_from_checkpoint(args.ckpt, map_location=device)

    splits = {
        'seen':      ('features_test_seen.pt',     'test_seen.csv'),
        'zeroshot':  ('features_test_zeroshot.pt', 'test_zeroshot.csv'),
    }

    for split_name, (feat_file, csv_file) in splits.items():
        feat_path = os.path.join(features_dir, feat_file)
        csv_path  = os.path.join(csv_dir, csv_file)

        print(f"\n[{split_name.upper()}] Loading split: {feat_path}")
        dataset = EpicKitchensFeatureDatasetWithMeta(feat_path, csv_path)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        all_video, all_text, meta = collect_embeddings(model, loader, device)
        print_top10_r1_errors(all_video, all_text, meta, split_name)


if __name__ == '__main__':
    main()