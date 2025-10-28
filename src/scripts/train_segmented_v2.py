# src/scripts/train_segmented_v2.py
from pathlib import Path
import sys, json, math
import numpy as np
import joblib
import librosa
from collections import Counter

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# Upewnij się, że widzimy repo root (jeśli chcesz korzystać z modułów z src/)
sys.path.append(str(Path(__file__).resolve().parents[2]))

RAW = Path("data/raw")
MODEL_OUT = Path("models/baseline_svc_segmented.joblib")
REPORT_OUT = Path("reports/metrics_segmented.json")

# --- Parametry segmentacji ---
SR = 22050
SEG_SECONDS = 10.0     # długość segmentu
HOP_SECONDS = 5.0      # przesunięcie między segmentami (50% overlap)
N_MFCC = 20
N_MELS = 128

def _stat(mat: np.ndarray) -> np.ndarray:
    return np.hstack([np.mean(mat,1), np.std(mat,1), np.median(mat,1), np.min(mat,1), np.max(mat,1)])

def extract_features_v15_segment(path: str, offset: float, duration: float) -> np.ndarray:
    """Wersja extract_features_v15 na fragment (offset/duration)."""
    y, sr = librosa.load(path, sr=SR, mono=True, offset=offset, duration=duration)

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))**2
    mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=N_MELS, fmin=20, fmax=sr//2)
    logmel = librosa.power_to_db(mel + 1e-10, ref=np.max)

    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S + 1e-10), n_mfcc=N_MFCC)
    d1   = librosa.feature.delta(mfcc)
    d2   = librosa.feature.delta(mfcc, order=2)

    chroma   = librosa.feature.chroma_stft(S=S, sr=sr)
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    rolloff  = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)
    zcr      = librosa.feature.zero_crossing_rate(y)
    rms      = librosa.feature.rms(S=S)

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

def segment_offsets(duration_s: float, seg: float, hop: float):
    if duration_s <= 0 or seg <= 0 or hop <= 0:
        return []
    offsets = []
    t = 0.0
    while t + seg <= duration_s + 1e-6:
        offsets.append(t)
        t += hop
    if not offsets and duration_s > 0:
        offsets = [0.0]
    return offsets

# --- Wczytanie i segmentacja danych ---
X, y, groups = [], [], []   # groups: id utworu, by segmenty jednego tracka były w tej samej grupie
file_count = 0

for label_dir in sorted(RAW.iterdir()):
    if not label_dir.is_dir():
        continue
    label = label_dir.name
    for f in sorted(label_dir.glob("*.wav")):
        file_count += 1
        track_id = f.stem  # np. jazz.00054
        try:
            dur = librosa.get_duration(path=str(f))
        except Exception:
            continue
        offs = segment_offsets(dur, SEG_SECONDS, HOP_SECONDS)
        for off in offs:
            try:
                feats = extract_features_v15_segment(str(f), offset=off, duration=SEG_SECONDS)
                X.append(feats); y.append(label); groups.append(track_id)
            except Exception:
                # jeśli segment na końcu nie działa, pomiń
                continue

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y)
groups = np.asarray(groups)

if len(X) == 0:
    raise SystemExit("Brak danych po segmentacji. Upewnij się, że w data/raw/ masz *.wav")

print(f"Utworów: {file_count} | Segmentów: {len(X)} | Cecha dim: {X.shape[1]}")

# --- Pipeline: scaler -> PCA(95%, whiten) -> SVC(RBF, balanced) ---
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95, whiten=True, random_state=0)),
    ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=0)),
])

# Mała siatka (12 kandydatów)
param_grid = {
    "clf__C":     [10, 30, 100, 300],
    "clf__gamma": [1e-4, 3e-4, 1e-3]
}

# GroupKFold: unikamy wycieku (segmenty tego samego utworu nie trafiają jednocześnie do train i test)
outer = GroupKFold(n_splits=5)
inner = GroupKFold(n_splits=3)

outer_acc, outer_f1, best_params_each = [], [], []

for fold, (tr_idx, te_idx) in enumerate(outer.split(X, y, groups), 1):
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]
    gtr = groups[tr_idx]

    gs = GridSearchCV(
        pipe, param_grid=param_grid, cv=inner.split(Xtr, ytr, gtr),
        n_jobs=-1, scoring="accuracy", refit=True, verbose=0
    )
    gs.fit(Xtr, ytr)

    ypred = gs.predict(Xte)
    acc = accuracy_score(yte, ypred)
    f1 = f1_score(yte, ypred, average="macro")

    outer_acc.append(acc)
    outer_f1.append(f1)
    best_params_each.append(gs.best_params_)

    print(f"[Fold {fold}] acc={acc:.4f}  f1_macro={f1:.4f} | best={gs.best_params_}")

acc_mean, acc_std = float(np.mean(outer_acc)), float(np.std(outer_acc))
f1_mean,  f1_std  = float(np.mean(outer_f1)),  float(np.std(outer_f1))
print(f"Outer-CV (GroupKFold): ACC {acc_mean:.4f}±{acc_std:.4f} | F1_macro {f1_mean:.4f}±{f1_std:.4f}")

# Refit na CAŁYM zbiorze z parametrami najczęściej wybieranymi
pair = Counter([tuple(sorted(p.items())) for p in best_params_each]).most_common(1)[0][0]
best_params = dict(pair)
print("Refit z parametrami:", best_params)

final_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95, whiten=True, random_state=0)),
    ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=0)),
])
final_pipe.set_params(**best_params)
final_pipe.fit(X, y)

MODEL_OUT.parent.mkdir(exist_ok=True)
joblib.dump(final_pipe, MODEL_OUT)
print(f"✅ Zapisano: {MODEL_OUT}")

REPORT_OUT.parent.mkdir(exist_ok=True)
with open(REPORT_OUT, "w", encoding="utf-8") as f:
    json.dump({
        "outer_cv": {
            "acc_mean": acc_mean, "acc_std": acc_std,
            "f1_macro_mean": f1_mean, "f1_macro_std": f1_std,
            "best_params_each_fold": best_params_each
        },
        "refit_params": best_params,
        "segmentation": {
            "sr": SR, "seg_seconds": SEG_SECONDS, "hop_seconds": HOP_SECONDS
        }
    }, f, ensure_ascii=False, indent=2)
print(f"📝 Raport: {REPORT_OUT}")
