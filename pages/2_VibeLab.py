# VibeMatch v1.5 — TOP5 bar chart + data augmentation + dataset cleaning
# ---------------------------------------------------------------
# Kluczowe zmiany vs v1.4
# 1) TOP5 bar chart po inferencji (matplotlib + st.pyplot)
# 2) Rozszerzanie puli danych przez augmentacje (pitch shift, time-stretch)
# 3) Czyszczenie danych: usunięcie plików "sine"/"square" i sygnałów nienaturalnych (heurystyki)
# ---------------------------------------------------------------

import os
import io
import glob
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import librosa.effects
import soundfile as sf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import streamlit as st

# (opcjonalnie) MLflow — logowanie playlist/nastrojów
try:
    import mlflow
    MLFLOW_OK = True
except Exception:
    MLFLOW_OK = False

warnings.filterwarnings("ignore")

# ----------------------------
# Konfiguracja aplikacji
# ----------------------------
st.set_page_config(page_title="VibeMatch v1.5", page_icon="🎵", layout="wide")
st.title("🎵 VibeMatch v1.5 — TOP5 chart & clean‑up")

# Ścieżka do danych (foldery klas, np. ./data/blues/*.wav etc.)
DATA_DIR = st.text_input("📁 Folder z danymi (foldery = gatunki)", value="./data")
SAMPLE_RATE = st.number_input("🎚️ Sample rate", value=22050, min_value=8000, max_value=48000, step=1000)
DURATION_SEC = st.number_input("⏱️ Minimalny czas nagrania [s]", value=2.0, min_value=0.5, max_value=30.0, step=0.5)

# Augmentacje — włącz/wyłącz + parametry
st.sidebar.header("🎛️ Augmentacje (rozszerz pulę danych)")
USE_AUG = st.sidebar.checkbox("Włącz augmentacje", value=False)
PITCH_STEPS = st.sidebar.multiselect("Pitch shift (półtony)", options=[-4, -2, -1, 1, 2, 4], default=[-2, 2])
TIME_STRETCH = st.sidebar.multiselect("Time‑stretch (x)", options=[0.9, 1.1, 0.8, 1.2], default=[0.9, 1.1])
MAX_AUG_PER_FILE = st.sidebar.slider("Maks. wariantów/plik", 0, 6, 2)

# Czyszczenie — włącz/wyłącz + progi
st.sidebar.header("🧹 Czyszczenie danych")
CLEAN_ENABLE = st.sidebar.checkbox("Włącz czyszczenie danych", value=True)
PEAK_DOMINANCE_MAX = st.sidebar.slider("Maks. dominacja jednej częstotliwości", 0.50, 0.99, 0.92, 0.01)
BANDWIDTH_MIN = st.sidebar.slider("Min. szerokość pasma (hz)", 100.0, 4000.0, 800.0, 50.0)
RMS_VAR_MIN = st.sidebar.slider("Min. wariancja RMS", 1e-6, 0.01, 1e-4)

# Model (prosty baseline — wstaw swój model jeśli masz):
@st.cache_data(show_spinner=False)
def build_model():
    # Prosty pipeline: standaryzacja + logreg (multinomial)
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=2000, multi_class="multinomial")),
    ])
    return clf

# ----------------------------
# Ekstrakcja cech (MFCC + statystyki)
# ----------------------------

def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    # Delta i Delta-Delta (opcjonalnie poprawia wyniki)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    # Statystyki po czasie (średnia + std)
    feats = [
        mfcc.mean(axis=1), mfcc.std(axis=1),
        d1.mean(axis=1), d1.std(axis=1),
        d2.mean(axis=1), d2.std(axis=1),
    ]
    feats = np.concatenate(feats, axis=0)
    return feats.astype(np.float32)

# ----------------------------
# Heurystyki czyszczenia sygnału
# ----------------------------

def is_file_name_bad(p: Path) -> bool:
    name = p.name.lower()
    return ("sine" in name) or ("square" in name) or ("saw" in name)


def audio_quality_ok(y: np.ndarray, sr: int) -> bool:
    """Zgrubne wykrywanie nienaturalnych sygnałów:
    - za mała zmienność RMS (ciągły ton)
    - zbyt wąskie pasmo (brak bogactwa częstotliwości)
    - dominacja jednej częstotliwości (peak_dominance)
    """
    if y is None or len(y) < sr * DURATION_SEC:
        return False

    # RMS
    rms = librosa.feature.rms(y=y)[0]
    rms_var = np.var(rms)
    if rms_var < RMS_VAR_MIN:
        return False

    # Szerokość pasma
    spec = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    # Spektralna szerokość pasma (approx)
    psd = spec.mean(axis=1)
    psd_sum = psd.sum() + 1e-12
    centroid = (psd * freqs).sum() / psd_sum
    var = (psd * (freqs - centroid) ** 2).sum() / psd_sum
    bandwidth = math.sqrt(max(var, 0.0))
    if bandwidth < BANDWIDTH_MIN:
        return False

    # Dominacja jednego piku
    peak = psd.max()
    peak_dominance = float(peak / (psd_sum + 1e-12))
    if peak_dominance > PEAK_DOMINANCE_MAX:
        return False

    return True

# ----------------------------
# Augmentacje
# ----------------------------

def augment_wave(y: np.ndarray, sr: int,
                 pitch_steps: list[int],
                 stretch_rates: list[float],
                 max_variants: int) -> list[np.ndarray]:
    """Generuje do 'max_variants' przekształceń: pitch shift i time‑stretch.
    Zwraca listę zsamplowanych wersji y (bez oryginału)."""
    if max_variants <= 0:
        return []

    variants = []

    # Naprzemiennie generuj pitch i stretch, aż do limitu
    for ps in pitch_steps:
        if len(variants) >= max_variants:
            break
        try:
            yp = librosa.effects.pitch_shift(y, sr=sr, n_steps=ps)
            variants.append(yp)
        except Exception:
            pass

    for rate in stretch_rates:
        if len(variants) >= max_variants:
            break
        if rate <= 0:
            continue
        try:
            ys = librosa.effects.time_stretch(y, rate=rate)
            # Ucinamy/wyśrodkowujemy do min. długości
            if len(ys) >= int(sr * DURATION_SEC):
                variants.append(ys)
        except Exception:
            pass

    return variants[:max_variants]

# ----------------------------
# Loader datasetu z czyszczeniem i opcjonalną augmentacją
# ----------------------------

def load_dataset(data_dir: str, sr: int,
                 use_augment: bool = False,
                 pitch_steps: list[int] = None,
                 stretch_rates: list[float] = None,
                 max_aug_per_file: int = 0,
                 clean_enable: bool = True):

    X, y, skipped = [], [], []

    class_dirs = sorted([p for p in Path(data_dir).glob("*") if p.is_dir()])
    class_names = [p.name for p in class_dirs]

    for cls_dir in class_dirs:
        label = cls_dir.name
        for wav in sorted(cls_dir.rglob("*.wav")):
            if clean_enable and is_file_name_bad(wav):
                skipped.append((str(wav), "bad_name"))
                continue

            try:
                sig, _ = librosa.load(wav, sr=sr, mono=True)
            except Exception:
                skipped.append((str(wav), "load_error"))
                continue

            if clean_enable and not audio_quality_ok(sig, sr):
                skipped.append((str(wav), "quality_fail"))
                continue

            if len(sig) < int(sr * DURATION_SEC):
                skipped.append((str(wav), "too_short"))
                continue

            # oryginał
            X.append(extract_features(sig, sr))
            y.append(label)

            # augmentacje
            if use_augment:
                aug_waves = augment_wave(sig, sr, pitch_steps or [], stretch_rates or [], max_aug_per_file)
                for aw in aug_waves:
                    if len(aw) >= int(sr * DURATION_SEC):
                        X.append(extract_features(aw, sr))
                        y.append(label)

    X = np.vstack(X) if len(X) else np.zeros((0, 120), dtype=np.float32)
    y = np.array(y)
    return X, y, class_names, skipped

# ----------------------------
# Trenowanie + raport
# ----------------------------

def train_and_report():
    with st.spinner("Ładowanie danych…"):
        X, y, class_names, skipped = load_dataset(
            DATA_DIR, SAMPLE_RATE,
            use_augment=USE_AUG,
            pitch_steps=PITCH_STEPS,
            stretch_rates=TIME_STRETCH,
            max_aug_per_file=MAX_AUG_PER_FILE,
            clean_enable=CLEAN_ENABLE,
        )

    st.write(f"Załadowano **{len(y)}** próbek po czyszczeniu i augmentacjach.")
    # Liczność klas po filtrach
    counts = pd.Series(y).value_counts().reindex(class_names, fill_value=0)
    st.caption("Liczność próbek w klasach (po filtrach i augmentacjach):")
    st.dataframe(counts.rename_axis("gatunek").reset_index(name="liczba"))
    if len(skipped):
        st.caption(f"Pominięto {len(skipped)} plików (np. 'sine/square', zbyt krótkie, zła jakość itp.)")
        if st.expander("Pokaż listę pominiętych plików").checkbox("Pokaż teraz"):
            st.write(pd.DataFrame(skipped, columns=["path", "reason"]))

    if len(y) < 20 or len(np.unique(y)) < 2:
        st.error("Za mało danych po filtrach. Poluzuj progi lub dodaj próbki.")
        return None, None, None

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = build_model()
    with st.spinner("Trenowanie modelu…"):
        model.fit(Xtr, ytr)

    ypred = model.predict(Xte)
    st.subheader("📋 Classification report")
    st.code(classification_report(yte, ypred, digits=3))

    # Confusion matrix (matplotlib, bez seaborn)
    cm = confusion_matrix(yte, ypred, labels=class_names)
    fig_cm, ax_cm = plt.subplots(figsize=(6.5, 5.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax_cm, cmap=None, colorbar=False)
    ax_cm.set_title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig_cm, clear_figure=True)

    return model, class_names, (Xte, yte)

# ----------------------------
# TOP5 bar chart
# ----------------------------

def plot_top5(proba: np.ndarray, class_names: list[str]):
    idx = np.argsort(proba)[::-1][:5]
    labels = [class_names[i] for i in idx]
    vals = proba[idx]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.barh(range(len(labels)), vals)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Prawdopodobieństwo")
    ax.set_title("TOP‑5 przewidywań")
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center")
    st.pyplot(fig, clear_figure=True)

# ----------------------------
# Mood Picker (tryb "stable" bez trenowania)
# ----------------------------

MOOD_GENRES = {
    "Calm": ["classical", "jazz"],
    "Focus": ["classical", "jazz"],
    "Chill": ["reggae", "blues"],
    "Happy": ["pop", "disco"],
    "Energy": ["hiphop", "metal"],
    "Night Drive": ["rock", "jazz"],
}

@st.cache_data(show_spinner=False)
def list_tracks_by_genres(data_dir: str, genres: list[str]) -> list[Path]:
    paths = []
    for g in genres:
        gdir = Path(data_dir) / g
        if gdir.exists():
            paths.extend(sorted(gdir.rglob("*.wav")))
    return paths


def est_bpm_rms(y: np.ndarray, sr: int) -> tuple[float, float]:
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    except Exception:
        tempo = np.nan
    rms = float(np.mean(librosa.feature.rms(y=y))) if y.size else np.nan
    return float(tempo), rms

st.subheader("🧭 Mood Picker (tryb prosty)")

# --- Prostą interpretacja intencji użytkownika (PL) -> mood ---
INTENT_KEYWORDS = {
    "Calm": ["spokój", "spokojny", "wyciszenie", "relaks", "uspokojenie", "medyt", "odpręż"],
    "Focus": ["skupienie", "koncentr", "nauka", "praca", "czytanie", "focus"],
    "Chill": ["zmęczony", "zmeczenie", "chill", "luźno", "luuz", "odpoczynek", "relaks"],
    "Happy": ["radość", "szczęśliwy", "szczescie", "wesolo", "dobry humor", "uśmiech", "happy"],
    "Energy": ["energia", "energetycznie", "power", "motywacja", "pobudzenie", "trening", "bieganie"],
    "Night Drive": ["noc", "nocna jazda", "night", "drive", "wieczór", "wieczor", "droga"]
}

def suggest_mood_from_text(text: str) -> str | None:
    if not text:
        return None
    t = text.lower()
    scores = {m: 0 for m in MOOD_GENRES}
    for mood_key, kws in INTENT_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                scores[mood_key] += 1
    # dodatkowe proste reguły
    if any(w in t for w in ["smut", "przygnęb", "nostalg", "zaduma"]):
        scores["Calm"] += 1; scores["Chill"] += 1
    if all(v == 0 for v in scores.values()):
        return None
    return max(scores, key=scores.get)

colm1, colm2, colm3 = st.columns([1,1,1])
with colm1:
    mood = st.selectbox("Wybierz nastrój", list(MOOD_GENRES.keys()), index=2)

# Pole na swobodny opis nastroju
user_intent = st.text_input("Opisz, na co masz ochotę (np. 'jestem zmęczony', 'rozpiera mnie energia', 'chcę się skupić')")
if st.button("💡 Podpowiedz nastrój na podstawie opisu"):
    smood = suggest_mood_from_text(user_intent)
    if smood:
        st.success(f"Proponowany nastrój: **{smood}** (na podstawie opisu)")
        mood = smood
    else:
        st.info("Nie rozumiem jeszcze tego opisu — wybierz nastrój z listy powyżej.")
with colm2:
    k_tracks = st.number_input("Ile utworów?", 1, 20, 8)
with colm3:
    use_energy = st.checkbox("Dopasuj energię (BPM/RMS)", value=False)

log_to_mlflow = st.checkbox("Loguj do MLflow (jeśli dostępne)", value=False, help="Wymaga skonfigurowanego MLflow (tracking URI)")

if st.button("🎲 Generuj playlistę"):
    target_genres = MOOD_GENRES[mood]
    files = list_tracks_by_genres(DATA_DIR, target_genres)
    if not files:
        st.warning("Brak plików dla wybranych gatunków. Sprawdź DATA_DIR.")
    else:
        # losowo + opcjonalne sortowanie po energii
        rng = np.random.default_rng(42)
        pick = files if len(files) <= 50 else list(rng.choice(files, size=50, replace=False))
        rows = []
        for p in pick:
            try:
                y, sr = librosa.load(p, sr=SAMPLE_RATE, mono=True)
                bpm, rms = est_bpm_rms(y, sr) if use_energy else (np.nan, np.nan)
                rows.append({"file": str(p), "genre": p.parent.name, "bpm": bpm, "rms": rms})
            except Exception:
                continue
        df = pd.DataFrame(rows)
        if use_energy and len(df):
            if mood in ("Calm", "Focus"):
                df = df.sort_values(["bpm", "rms"], ascending=[True, True])
            else:
                df = df.sort_values(["bpm", "rms"], ascending=[False, False])
        playlist = df.head(int(k_tracks)) if len(df) else df
        st.success(f"Wygenerowano playlistę dla nastroju **{mood}**")
        st.dataframe(playlist[["genre", "bpm", "rms", "file"]])

        # mały podgląd audio (pierwsze 1-2 utwory)
        for i, row in playlist.head(2).iterrows():
            try:
                with open(row["file"], "rb") as f:
                    st.audio(f.read())
            except Exception:
                pass

        # MLflow logging (opcjonalnie)
        if log_to_mlflow and MLFLOW_OK and len(playlist):
            try:
                with mlflow.start_run(run_name=f"mood_{mood}"):
                    mlflow.log_param("mood", mood)
                    mlflow.log_param("genres", ",".join(target_genres))
                    mlflow.log_param("k_tracks", int(k_tracks))
                    mlflow.log_param("use_energy", use_energy)
                    if use_energy:
                        mlflow.log_metric("avg_bpm", float(pd.to_numeric(playlist["bpm"], errors="coerce").mean()))
                        mlflow.log_metric("avg_rms", float(pd.to_numeric(playlist["rms"], errors="coerce").mean()))
                    # zapisz playlistę jako artifact
                    out_csv = Path("playlist_mood.csv")
                    playlist.to_csv(out_csv, index=False)
                    mlflow.log_artifact(str(out_csv))
                st.caption("Zalogowano do MLflow (parametry, metryki, artifact=CSV)")
            except Exception as e:
                st.warning(f"Nie udało się zalogować do MLflow: {e}")

# ----------------------------
# Sekcja trenowania i inferencji
# ----------------------------

if st.button("🚀 Wytrenuj model / Odśwież"):
    st.session_state["trained"] = True
    st.session_state["model"], st.session_state["class_names"], st.session_state["testpack"] = train_and_report()

if st.session_state.get("model") is not None:
    st.divider()
    st.subheader("🎧 Szybka inferencja na pliku WAV")
    up = st.file_uploader("Wrzuć plik .wav", type=["wav"])
    if up is not None:
        data, sr = sf.read(io.BytesIO(up.read()))
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SAMPLE_RATE:
            data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        feats = extract_features(data.astype(np.float32), sr).reshape(1, -1)
        proba = st.session_state["model"].predict_proba(feats)[0]
        pred_idx = int(np.argmax(proba))
        pred = st.session_state["class_names"][pred_idx]

        st.success(f"🎯 Predykcja: **{pred}**")
        plot_top5(proba, st.session_state["class_names"])
