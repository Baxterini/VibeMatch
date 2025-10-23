# src/features.py
from pathlib import Path
import numpy as np, librosa

def extract_features(path: str, sr_target=22050, n_mfcc=20) -> np.ndarray:
    y, sr = librosa.load(path, sr=sr_target, mono=True)
    # Podstawy: MFCC + delta + deltadelta
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    feat = np.vstack([mfcc, d1, d2]).mean(axis=1)
    # RMS (energia)
    rms = librosa.feature.rms(y=y).mean(axis=1)
    return np.concatenate([feat, rms])
