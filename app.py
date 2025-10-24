# app.py
import os, tempfile, numpy as np, joblib, streamlit as st

# --- importy cech dla obu wersji ---
from src.features import extract_features as extract_v1
try:
    from src.features_v15 import extract_features_v15 as extract_v15
except Exception:
    extract_v15 = None  # jeśli jeszcze nie masz pliku v1.5

st.set_page_config(page_title="VibeMatch Tech Lab", page_icon="🎶")
st.title("🎶 VibeMatch – baseline")

# --- wybór modelu: najpierw v1.5, potem v1.0 ---
MODEL_PATHS = [
    ("models/baseline_v1_5.joblib", "v1.5"),
    ("models/baseline_logreg.joblib", "v1.0"),
]

clf, model_ver = None, None
for mp, ver in MODEL_PATHS:
    if os.path.exists(mp):
        clf = joblib.load(mp)
        model_ver = ver
        break

if clf is None:
    st.warning("Nie znalazłem modelu. Uruchom trening: `python -m scripts.train_v15` (v1.5) lub `python -m scripts.train_baseline` (v1.0).")
    st.stop()

# --- dobór funkcji cech do wersji modelu ---
if model_ver == "v1.5":
    if extract_v15 is None:
        st.error("Model v1.5 wymaga `src/features_v15.py`. Utwórz go zgodnie z instrukcją i spróbuj ponownie.")
        st.stop()
    feature_fn = extract_v15
else:
    feature_fn = extract_v1

st.caption(f"Załadowano model: **{model_ver}**")

file = st.file_uploader("Wrzuć plik .wav", type=["wav"])
if file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file.read())
        wav_path = tmp.name

    # odtwarzacz audio
    with open(wav_path, "rb") as f:
        st.audio(f.read(), format="audio/wav")

    # ekstrakcja cech + predykcja
    feats = feature_fn(wav_path).reshape(1, -1)
    proba = clf.predict_proba(feats)[0]
    pred = clf.classes_[np.argmax(proba)]

    st.subheader(f"Predykcja: **{pred}**")
    # ładniejsze wypisanie prob
    proba_pairs = sorted(zip(clf.classes_, proba), key=lambda t: t[1], reverse=True)
    st.write({cls: float(p) for cls, p in proba_pairs})
