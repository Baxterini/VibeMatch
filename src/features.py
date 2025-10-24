import numpy as np
import librosa

def _safe_stat(mat: np.ndarray) -> np.ndarray:
    return np.hstack([
        np.mean(mat, axis=1),
        np.std(mat, axis=1),
        np.median(mat, axis=1),
        np.min(mat, axis=1),
        np.max(mat, axis=1),
    ])

def extract_features(path: str, sr_target: int = 22050, n_mfcc: int = 20) -> np.ndarray:
    y, sr = librosa.load(path, sr=sr_target, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    d1   = librosa.feature.delta(mfcc)
    d2   = librosa.feature.delta(mfcc, order=2)

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))**2
    chroma   = librosa.feature.chroma_stft(S=S, sr=sr)

    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    rolloff  = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)

    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(S=S)

    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo   = float(np.atleast_1d(tempo)[0])
        n_beats = int(len(beats)) if beats is not None else 0
    except Exception:
        tempo, n_beats = 0.0, 0
    tempo_vec = np.array([tempo, n_beats], dtype=np.float32)

    feats = np.hstack([
        _safe_stat(mfcc),
        _safe_stat(d1),
        _safe_stat(d2),
        _safe_stat(chroma),
        _safe_stat(centroid),
        _safe_stat(rolloff),
        _safe_stat(zcr),
        _safe_stat(rms),
        tempo_vec.flatten(),
    ])
    return feats.astype(np.float32)
