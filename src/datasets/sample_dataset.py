import pandas as pd
import numpy as np

np.random.seed(42)

def select_zeroshot_classes(train_series, val_series, exclude_top_n=15, min_val_samples=15, num_to_select=4):
    train_counts = train_series.value_counts()
    val_counts = val_series.value_counts()
    
    top_train_classes = train_counts.head(exclude_top_n).index.tolist()
    
    robust_val_classes = val_counts[val_counts >= min_val_samples].index.tolist()
    
    candidates = [c for c in robust_val_classes if c not in top_train_classes]
    
    return candidates[:num_to_select]

print("--- Loading original annotations for training and validation ---")
train_csv_path = './data/annotations/raw/EPIC_100_train.csv' 
val_csv_path = './data/annotations/raw/EPIC_100_validation.csv'

df_train = pd.read_csv(train_csv_path)
df_val = pd.read_csv(val_csv_path)

print("\n--- Zero-shot classes selection ---")

zero_shot_verbs = select_zeroshot_classes(df_train['verb'], df_val['verb'], exclude_top_n=15, min_val_samples=20, num_to_select=4)
zero_shot_nouns = select_zeroshot_classes(df_train['noun'], df_val['noun'], exclude_top_n=20, min_val_samples=20, num_to_select=4)

print(f"Zero-shot verbs selected: {zero_shot_verbs}")
print(f"Zero-shot nouns selected: {zero_shot_nouns}")

val_zs_verb_count = df_val[df_val['verb'].isin(zero_shot_verbs)].shape[0]
val_zs_noun_count = df_val[df_val['noun'].isin(zero_shot_nouns)].shape[0]
print(f"Number of samples in validation for zero-shot verbs: {val_zs_verb_count}")
print(f"Number of samples in validation for zero-shot nouns: {val_zs_noun_count}")


print("\n--- Excluding selected classes from the training set ---")
mask_train_safe = (
    ~df_train['verb'].isin(zero_shot_verbs) & 
    ~df_train['noun'].isin(zero_shot_nouns)
)

df_train_safe = df_train[mask_train_safe]

print("\n--- Sampling 300 video from the training set ---")
unique_safe_video_ids = df_train_safe['video_id'].unique()

sampled_video_ids = np.random.choice(unique_safe_video_ids, size=300, replace=False)
    
df_train_final = df_train_safe[df_train_safe['video_id'].isin(sampled_video_ids)]

print(f"Total number of samples in the final training set: {len(df_train_final)}")

print("\n--- Validation set split in seen and zero-shot ---")
mask_val_zeroshot = (
    df_val['verb'].isin(zero_shot_verbs) | 
    df_val['noun'].isin(zero_shot_nouns)
)
df_val_zeroshot = df_val[mask_val_zeroshot]
df_val_seen = df_val[~mask_val_zeroshot]

print(f"Number of samples in seen validation set: {len(df_val_seen)}")
print(f"Number of samples in zero-shot validation set: {len(df_val_zeroshot)}")

print("\n--- Saving files ---")
df_train_final.to_csv('./data/annotations/processed/train.csv', index=False)
df_val_seen.to_csv('./data/annotations/processed/val_seen.csv', index=False)
df_val_zeroshot.to_csv('./data/annotations/processed/val_zeroshot.csv', index=False)
print("\n--- Process completed successfully ---")