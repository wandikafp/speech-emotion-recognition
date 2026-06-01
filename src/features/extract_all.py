"""
Extract features from RAVDESS dataset for training.
Applies offline data augmentation to the training set to prevent overfitting
and match ETASR paper targets (>90% accuracy).
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# Ensure we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features.extractor import preprocess_audio, extract_features
from src.features.augment import augment

FEAT_DIR = "data/processed/features"
MANIFEST = "data/processed/manifest.csv"

# Augmentation multiplier: how many augmented versions to generate per train sample
AUGMENT_COPIES = 4

def main():
    df = pd.read_csv(MANIFEST)
    emotion_classes = sorted(df['emotion_label'].unique())
    label2idx = {e: i for i, e in enumerate(emotion_classes)}

    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split].reset_index(drop=True)
        out_dir  = os.path.join(FEAT_DIR, split)
        os.makedirs(out_dir, exist_ok=True)
        scaler_path = os.path.join(FEAT_DIR, 'train', 'scaler.pkl')

        X_list, y_list = [], []
        print(f"\n── Extracting {split} split ({len(split_df)} base files) ──")

        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            y_audio = preprocess_audio(row['file_path'])
            label_idx = label2idx[row['emotion_label']]
            
            # 1. Clean feature (always extracted)
            X_list.append(extract_features(y_audio))
            y_list.append(label_idx)

            # 2. Augmented features (only for training)
            if split == 'train':
                for _ in range(AUGMENT_COPIES):
                    y_aug = augment(y_audio)
                    X_list.append(extract_features(y_aug))
                    y_list.append(label_idx)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list,  dtype=np.int32)
        print(f"\n  Raw shape: X={X.shape}, y={y.shape}")

        N, T, F = X.shape
        X_flat = X.reshape(-1, F)

        if split == 'train':
            scaler = StandardScaler().fit(X_flat)
            joblib.dump(scaler, scaler_path)
            print(f"  Saved scaler -> {scaler_path}")
        else:
            scaler = joblib.load(scaler_path)

        X_norm = scaler.transform(X_flat).reshape(N, T, F).astype(np.float32)
        np.save(os.path.join(out_dir, 'X.npy'), X_norm)
        np.save(os.path.join(out_dir, 'y.npy'), y)
        joblib.dump({'label2idx': label2idx, 'idx2label': {v:k for k,v in label2idx.items()}},
                    os.path.join(out_dir, 'label_encoder.pkl'))
        
        print(f"  ✅ Saved normalized features -> {out_dir}")

if __name__ == "__main__":
    main()
