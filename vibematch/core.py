from functools import lru_cache
from pathlib import Path
import sys
import numpy as np
import joblib

# najpierw upewnij się, że root repo jest w sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features_v15 import extract_features_v15 as extract_features

# preferuj model segmentowany, fallback do starego
MODEL_CANDIDATES = [
    Path("models/baseline_svc_segmented.joblib"),
    Path("models/baseline_svc.joblib"),
]

def _pick_model_path():
    for p in MODEL_CANDIDATES:
        if p.exists():
            return p
    return MODEL_CANDIDATES[-1]

MODEL_PATH = _pick_model_path()

@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

GENRE_TO_MOOD = {
    "rock": "energy", "metal": "energy", "disco": "party", "hiphop": "party",
    "classical": "focus", "jazz": "chill", "blues": "sad",
    "pop": "happy", "reggae": "happy", "country": "happy",
}

def predict_genre(audio_path: str):
    """Zwraca: (pred_class, confidence, classes, probs)."""
    x = extract_features(audio_path)
    x = np.asarray(x, dtype=float).reshape(1, -1)

    model = load_model()
    if model is None:
        raise FileNotFoundError(f"Brak modelu: {MODEL_PATH}")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[0]
        classes = model.classes_
        i = int(np.argmax(probs))
        return str(classes[i]), float(probs[i]), classes, probs
    else:
        pred = model.predict(x)[0]
        return str(pred), 1.0, np.array([pred]), np.array([1.0])

def map_genre_to_mood(genre: str) -> str:
    return GENRE_TO_MOOD.get(str(genre).lower(), "chill")
