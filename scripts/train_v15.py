from pathlib import Path
import sys, numpy as np, joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# import projektu
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.features_v15 import extract_features_v15

RAW = Path("data/raw")
X, y = [], []
for label_dir in RAW.iterdir():
    if label_dir.is_dir():
        label = label_dir.name
        for f in label_dir.glob("*.wav"):
            X.append(extract_features_v15(str(f)))
            y.append(label)

if not X:
    raise SystemExit("Brak plików w data/raw/<klasa>/*.wav")

X = np.array(X); y = np.array(y)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipe = Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True))])
param_grid = {"clf__C":[1,3,10], "clf__gamma":["scale", 0.01, 0.001]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gs = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1, verbose=0)
gs.fit(Xtr, ytr)

acc_cv = gs.best_score_
acc_te = gs.score(Xte, yte)
print(f"Best CV acc: {acc_cv:.4f} | Test acc: {acc_te:.4f}")
print("Best params:", gs.best_params_)

# raport tekstowy
y_pred = gs.predict(Xte)
print("\nConfusion matrix:\n", confusion_matrix(yte, y_pred))
print("\nClassification report:\n", classification_report(yte, y_pred, digits=3))

Path("models").mkdir(exist_ok=True)
out_path = "models/baseline_v1_5.joblib"
joblib.dump(gs.best_estimator_, out_path)
print("Zapisano:", out_path)
