import os
import tarfile
import pandas as pd
from tqdm import tqdm
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.sampling_utils import get_uniform_frame_indices

def extract_required_frames(csv_path, dataset_root_dir, output_root_dir, strategy='center'):
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    grouped_videos = df.groupby('video_id')
    
    os.makedirs(output_root_dir, exist_ok=True)
    
    print(f"Starting to process {len(grouped_videos)} unique videos...")
    
    for video_id, group in tqdm(grouped_videos, desc="Processing videos - extracting and sampling frames", unit="video"):
        
        frames_to_extract = set()
        for _, row in group.iterrows():
            indices = get_uniform_frame_indices(
                row['start_frame'], 
                row['stop_frame'], 
                num_frames=8, 
                strategy=strategy
            )
            frames_to_extract.update(indices)
            
        target_filenames = {f"frame_{idx:010d}.jpg" for idx in frames_to_extract}
        
        participant_id = video_id.split('_')[0]
        tar_path = os.path.join(dataset_root_dir, participant_id, "rgb_frames", f"{video_id}.tar")
        
        if not os.path.exists(tar_path):
            print(f"\nFile not found: {tar_path}. Skipping video.")
            continue
            
        video_out_dir = os.path.join(output_root_dir, video_id)
        os.makedirs(video_out_dir, exist_ok=True)
        
        with tarfile.open(tar_path, 'r') as tar:
            members = tar.getmembers()
            for member in members:
                filename = os.path.basename(member.name)
                
                if filename in target_filenames:
                    member.name = filename 
                    tar.extract(member, path=video_out_dir)

if __name__ == "__main__":
    DATASET_ROOT = "./data/raw/epic_Kitchen100" 
    
    OUTPUT_ROOT = "./data/sampled/"

    for s in ["train", 'val', "test_seen", "test_zeroshot"]:
        output_dir = os.path.join(OUTPUT_ROOT, s)
        os.makedirs(output_dir, exist_ok=True)
    
        csv_to_process = os.path.join('./data/annotations/processed/', s + '.csv')
        
        extract_required_frames(
            csv_path=csv_to_process,
            dataset_root_dir=DATASET_ROOT,
            output_root_dir=output_dir,
            strategy='center' #center for offline feature extraction
        )
    
    print("Uniform temporal sampling completed successfully.")