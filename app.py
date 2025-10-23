import streamlit as st, numpy as np, tempfile, joblib
from src.features import extract_features

st.set_page_config(page_title="VibeMatch Tech Lab", page_icon="🎶")
st.title("🎶 VibeMatch – baseline")

clf = joblib.load("models/baseline_logreg.joblib")

file = st.file_uploader("Wrzuć plik .wav", type=["wav"])
if file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file.read())
        path = tmp.name
    feats = extract_features(path).reshape(1, -1)
    proba = clf.predict_proba(feats)[0]
    pred = clf.classes_[np.argmax(proba)]
    st.subheader(f"Predykcja: **{pred}**")
    st.write({cls: float(p) for cls, p in zip(clf.classes_, proba)})
