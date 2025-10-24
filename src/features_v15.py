import numpy as np
import librosa

def _stat(mat: np.ndarray) -> np.ndarray:
    return np.hstack([np.mean(mat,1), np.std(mat,1), np.median(mat,1), np.min(mat,1), np.max(mat,1)])

def extract_features_v15(path: str, sr_target: int = 22050, n_mfcc: int = 20, n_mels: int = 128) -> np.ndarray:
    y, sr = librosa.load(path, sr=sr_target, mono=True)

    # log-mel
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))**2
    mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=n_mels, fmin=20, fmax=sr//2)
    logmel = librosa.power_to_db(mel + 1e-10, ref=np.max)

    # MFCC + delty
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S + 1e-10), n_mfcc=n_mfcc)
    d1   = librosa.feature.delta(mfcc)
    d2   = librosa.feature.delta(mfcc, order=2)

    # Chroma / spektralne / ZCR / RMS
    chroma   = librosa.feature.chroma_stft(S=S, sr=sr)
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    rolloff  = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)
    zcr      = librosa.feature.zero_crossing_rate(y)
    rms      = librosa.feature.rms(S=S)

    # Tempo (odporne)
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.atleast_1d(tempo)[0])
        n_beats = int(len(beats)) if beats is not None else 0
    except Exception:
        tempo, n_beats = 0.0, 0
    tempo_vec = np.array([tempo, n_beats], dtype=np.float32)

    feats = np.hstack([
        _stat(logmel),
        _stat(mfcc), _stat(d1), _stat(d2),
        _stat(chroma), _stat(centroid), _stat(rolloff),
        _stat(zcr), _stat(rms),
        tempo_vec.flatten()
    ])
    return feats.astype(np.float32)
