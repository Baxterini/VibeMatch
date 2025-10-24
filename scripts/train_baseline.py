# scripts/train_baseline.py
from pathlib import Path
import sys
# dopisz katalog projektu do sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# scripts/train_baseline.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np, joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from src.features import extract_features

RAW = Path("data/raw")
X, y = [], []
for label_dir in RAW.iterdir():
    if label_dir.is_dir():
        label = label_dir.name
        for f in label_dir.glob("*.wav"):
            X.append(extract_features(str(f)))
            y.append(label)

if not X:
    raise SystemExit("Brak plików w data/raw/<klasa>/*.wav")

X = np.array(X); y = np.array(y)

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", probability=True))
])

param_grid = {
    "clf__C":   [1, 3, 10],
    "clf__gamma": ["scale", 0.01, 0.001]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gs = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1, verbose=0)
gs.fit(Xtr, ytr)

acc_tr = gs.best_score_
acc_te = gs.score(Xte, yte)
print(f"Best CV acc: {acc_tr:.4f}  |  Test acc: {acc_te:.4f}")
print("Best params:", gs.best_params_)

Path("models").mkdir(exist_ok=True)
joblib.dump(gs.best_estimator_, "models/baseline_logreg.joblib")  # zostawiamy tę samą ścieżkę
print("Zapisano: models/baseline_logreg.joblib")
