# src/scripts/train_baseline.py
from pathlib import Path
import sys, json
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ścieżka do root repo (żeby import z src/ działał)
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.features_v15 import extract_features_v15

RAW = Path("data/raw")
MODEL_OUT = Path("models/baseline_svc.joblib")
REPORT_OUT = Path("reports/metrics.json")

# 1) Wczytanie danych -> X, y
X, y = [], []
for label_dir in RAW.iterdir():
    if label_dir.is_dir():
        label = label_dir.name
        for f in label_dir.glob("*.wav"):
            X.append(extract_features_v15(str(f)))
            y.append(label)

if not X:
    raise SystemExit("Brak plików w data/raw/<klasa>/*.wav")

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y)

# 2) Pipeline: Skalowanie -> PCA(95%, whiten) -> SVC(RBF, balanced)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95, whiten=True, random_state=0)),
    ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=0)),
])

# 3) Mała, logarytmiczna siatka (16 kandydatów)
param_grid = {
    "clf__C":     np.logspace(-1, 2, 4),      # 0.1, 1, 10, 100
    "clf__gamma": np.logspace(-4, -1, 4),     # 1e-4, 1e-3, 1e-2, 1e-1
}

# 4) Zagnieżdżona CV: inner=3-fold do strojenia, outer=5-fold do oceny
inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)

outer_acc, outer_f1 = [], []
best_params_across = []

for fold, (tr_idx, te_idx) in enumerate(outer.split(X, y), 1):
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]

    gs = GridSearchCV(
        pipe, param_grid=param_grid, cv=inner, n_jobs=-1,
        scoring="accuracy", refit=True, verbose=0
    )
    gs.fit(Xtr, ytr)

    ypred = gs.predict(Xte)
    acc = accuracy_score(yte, ypred)
    f1 = f1_score(yte, ypred, average="macro")
    outer_acc.append(acc)
    outer_f1.append(f1)
    best_params_across.append(gs.best_params_)

    print(f"[Fold {fold}] acc={acc:.4f}  f1_macro={f1:.4f} | best={gs.best_params_}")

# 5) Raport ze średnią i odchyleniem
acc_mean, acc_std = float(np.mean(outer_acc)), float(np.std(outer_acc))
f1_mean,  f1_std  = float(np.mean(outer_f1)),  float(np.std(outer_f1))
print(f"Outer-CV: ACC {acc_mean:.4f}±{acc_std:.4f} | F1_macro {f1_mean:.4f}±{f1_std:.4f}")

# 6) Refit na CAŁYM zbiorze z parametrami, które najczęściej wygrywały
# (prosty wybór „modu best_params”)
from collections import Counter
pair = Counter([tuple(sorted(p.items())) for p in best_params_across]).most_common(1)[0][0]
best_params = dict(pair)
print("Refit na pełnym zbiorze z parametrami:", best_params)

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

# 7) Zapis metryk do raportu
REPORT_OUT.parent.mkdir(exist_ok=True)
with open(REPORT_OUT, "w", encoding="utf-8") as f:
    json.dump({
        "outer_cv": {
            "acc_mean": acc_mean, "acc_std": acc_std,
            "f1_macro_mean": f1_mean, "f1_macro_std": f1_std,
            "folds": [{"acc": float(a), "f1_macro": float(b)} for a, b in zip(outer_acc, outer_f1)],
            "best_params_each_fold": best_params_across,
        },
        "refit_params": best_params
    }, f, ensure_ascii=False, indent=2)
print(f"📝 Raport: {REPORT_OUT}")
