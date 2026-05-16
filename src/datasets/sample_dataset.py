import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
test_csv_path = './data/annotations/raw/EPIC_100_validation.csv' #in EPIC-KITCHEN-100, the test set is private


df_train = pd.read_csv(train_csv_path)
df_test = pd.read_csv(test_csv_path)

print("\n--- Zero-shot classes selection ---")

zero_shot_verbs = select_zeroshot_classes(df_train['verb'], df_test['verb'], exclude_top_n=15, min_val_samples=20, num_to_select=4)
zero_shot_nouns = select_zeroshot_classes(df_train['noun'], df_test['noun'], exclude_top_n=20, min_val_samples=20, num_to_select=4)

print(f"Zero-shot verbs selected: {zero_shot_verbs}")
print(f"Zero-shot nouns selected: {zero_shot_nouns}")

val_zs_verb_count = df_test[df_test['verb'].isin(zero_shot_verbs)].shape[0]
val_zs_noun_count = df_test[df_test['noun'].isin(zero_shot_nouns)].shape[0]
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
    
df_train_sampled = df_train_safe[df_train_safe['video_id'].isin(sampled_video_ids)]

print(f"Total number of samples in the final training set: {len(df_train_sampled)}")

print("\n--- Training set split in train and val 85%-15% ---")
unique_video_ids = df_train_sampled['video_id'].unique()
train_video_ids, val_video_ids = train_test_split(unique_video_ids, test_size=0.15, random_state=42)

df_train_final = df_train_sampled[df_train_sampled['video_id'].isin(train_video_ids)]
df_val_final = df_train_sampled[df_train_sampled['video_id'].isin(val_video_ids)]

print("\n--- Test set split in seen and zero-shot ---")
mask_test_zeroshot = (
    df_test['verb'].isin(zero_shot_verbs) | 
    df_test['noun'].isin(zero_shot_nouns)
)
df_test_zeroshot = df_test[mask_test_zeroshot]
df_test_seen = df_test[~mask_test_zeroshot]

print("\n--- Statistics ---")
print(f"Number of samples in train set: {len(df_train_final)}")
print(f"Number of samples in val set: {len(df_val_final)}")
print(f"Number of samples in seen test set: {len(df_test_seen)}")
print(f"Number of samples in zero-shot test set: {len(df_test_zeroshot)}")

print("\n--- Saving files ---")
df_train_final.to_csv('./data/annotations/processed/train.csv', index=False)
df_val_final.to_csv('./data/annotations/processed/val.csv', index=False)
df_test_seen.to_csv('./data/annotations/processed/test_seen.csv', index=False)
df_test_zeroshot.to_csv('./data/annotations/processed/test_zeroshot.csv', index=False)
print("\n--- Process completed successfully ---")