# src/train_baseline.py
from pathlib import Path
import numpy as np, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.features import extract_features

# Załóżmy strukturę: data/raw/<label>/*.wav
RAW = Path("data/raw")
X, y = [], []
for label_dir in RAW.iterdir():
    if label_dir.is_dir():
        label = label_dir.name
        for f in label_dir.glob("*.wav"):
            X.append(extract_features(str(f)))
            y.append(label)

X = np.array(X); y = np.array(y)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, n_jobs=None))
])

pipe.fit(Xtr, ytr)
acc = pipe.score(Xte, yte)
print("Baseline accuracy:", round(acc, 4))

Path("models").mkdir(exist_ok=True)
joblib.dump(pipe, "models/baseline_logreg.joblib")
