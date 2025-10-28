# app.py
import streamlit as st
st.set_page_config(page_title="VibeMatch • Audio Analyzer", page_icon="🎧", layout="centered")

import tempfile
import numpy as np
from datetime import datetime
from pathlib import Path
import re
import uuid
import pandas as pd
import os, glob, random
import urllib.parse as _url
import re, difflib

from vibematch.ui import apply_dark_base, apply_mood_background, render_equalizer
from vibematch.core import load_model, predict_genre, map_genre_to_mood

# --- Feedback helpers ---
FEEDBACK_PATH = Path("data/feedback.csv")
FEEDBACK_PATH.parent.mkdir(exist_ok=True)

def is_valid_email(x: str) -> bool:
    if not x:
        return True  # pole opcjonalne
    return re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", x) is not None

def save_feedback(row: dict) -> None:
    df = pd.DataFrame([row])
    if FEEDBACK_PATH.exists():
        df.to_csv(FEEDBACK_PATH, mode="a", header=False, index=False, encoding="utf-8")
    else:
        df.to_csv(FEEDBACK_PATH, index=False, encoding="utf-8")

# --- UI base ---
apply_dark_base()
st.title("🎧 VibeMatch — Audio Analyzer")
st.caption("Wrzuć krótki fragment muzyki (WAV/MP3/OGG), a powiem Ci, jaki to klimat.")

# --- VIBE FIRST: mood -> demo utwór + natychmiastowe tło ---
GTZAN_CLASSES = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

MOOD_TO_GTZAN = {
    "chill": "jazz",
    "focus": "classical",
    "energy": "rock",
    "happy": "pop",
    "party": "disco",
    "zen": "classical",
    "melancholy": "blues",
    "confidence": "hiphop",
    "romance": "jazz",
    "sunny": "reggae",
}


# 1) słownik słów-kluczy (PL/EN/emoji) -> mood
_MOOD_KEYWORDS = {
    # chill / relax
    "chill": [
        "chill","relax","relaks","luźno","wyluz","odpręż","odpoczynek","spokojnie","na luzie",
        "wieczornie","kojąco","kojenie","strefa komfortu","slow","calm","laid back","lofi",
        "🌊","😌","🧘","🌙","🫖","🛋️"
    ],

    # focus / study / deep work
    "focus": [
        "focus","fokus","skup","koncentr","koncentracja","pracować","praca","coding","kodowanie",
        "programować","nauka","uczyć","czytanie","czytać","deep work","bez rozpraszaczy","bez rozpraszania",
        "flow","pomodoro","study","library","biblioteka","quiet","silent",
        "📚","💻","⌨️","🎯","🔕","🧠"
    ],

    # energy / workout / dance-party
    "energy": [
        "energia","energetycznie","power","moc","boost","pobudka","nakręcić","nakręcony","drive","napęd",
        "trening","ćwiczyć","siłownia","gym","cardio","biegać","bieg","rower","bike","sprint",
        "taniec","tańczyć","dance","zumba","skakać","jump","rozrywka","rozkręcić","banger","bangier",
        "party hard","mocny bit","dopamina","kopa","pump",
        "🏃","🏋️","🚴","⚡","🔥","🕺","💃"
    ],

    # happy / feel-good
    "happy": [
        "szczęście","szczęśliwy","radość","radosny","uśmiech","pozytywnie","feel good","good vibes",
        "wesoło","euforia","euforycznie","lekko","pogodnie","słońce","sunny",
        "yay","yay!","super","fajnie","nice","miło",
        "😊","😄","😁","🌞","🥳","✨"
    ],

    # party / club / fun (rozrywka)
    "party": [
        "party","impreza","imprezowo","klub","club","parkiet","dancefloor","disco","nu-disco",
        "rozrywka","zabawa","bawić","balet","after","before","domówka","event",
        "dj","set","bity","głośno","bas","bass","wkręca",
        "🎉","🎈","🍾","🪩","💃","🕺","🎧"
    ],

    # zen / meditation / ambient
    "zen": [
        "zen","medyt","medytacja","mindfulness","uważność","oddech","oddychanie","spokojny","spokój",
        "kojący","wyciszenie","relaksacja","relaksacyjnie","ambient","white noise","szum","deszcz",
        "koncentracja oddech","yoga","joga","asana","savasana",
        "🧘","🌿","🕯️","🌫️","🌧️","🧖"
    ],

    # melancholy / nostalgia / rain
    "melancholy": [
        "smutek","smutno","nostalgia","nostalg","melancholia","melan","zaduma","zadumany","refleksja",
        "jesiennie","jesień","deszcz","deszczowo","plucha","ciemno","ciemność","minor","ballada",
        "łagodnie smutne","łezka","łzy","łza","tęsknota","tęsknić",
        "😢","🥀","🌧️","☔","🌫️","🌙"
    ],

    # confidence / motivation / hustle
    "confidence": [
        "pewność","pewny siebie","motywacja","motyw","determinacja","ambicja","ambitnie","wznoszę się",
        "przełamać","działać","działam","produktywnie","progres","wyzwanie","tryhard","hustle",
        "alpha","grind","napierać","idę po swoje","focus mode",
        "💪","🔥","🚀","🏆","📈","🛡️"
    ],

    # romance / love / cozy evening
    "romance": [
        "romans","romantycznie","romantyczny","miłość","kocham","love","randka","randkowo","kolacja",
        "świece","kwiaty","róże","nastrojowo","delikatnie","przytulnie","cozy","wieczór we dwoje",
        "slow dance","ballad",
        "❤️","💞","💖","🌹","🍷","🌅"
    ],

    # sunny / reggae / beach
    "sunny": [
        "słońce","słonecznie","plaża","lato","letnio","wakacje","beztrosko","palmy","piasek","surf",
        "rasta","reggae","roots","skanking","jamajka","tropiki","tropikalnie","kokos",
        "☀️","🏖️","🌴","🌺","🕶️","🍹"
    ],
}


# 2) heurystyka rozpoznania z tolerancją literówek (bez dodatkowych bibliotek)
def _mood_from_text(txt: str) -> str | None:
    if not txt:
        return None
    t = txt.lower().strip()

    # szybka ścieżka: słowo/emoji zawarte wprost
    for mood, keys in _MOOD_KEYWORDS.items():
        if any(k in t for k in keys):
            return mood

    # tolerancja literówek: spróbuj dopasować najbliższe słowo
    tokens = re.findall(r"[a-ząćęłńóśźż0-9\U0001F300-\U0001FAFF]+", t)
    for tok in tokens:
        for mood, keys in _MOOD_KEYWORDS.items():
            # szukamy najlepszego podobieństwa do któregoś z kluczy
            best = max((difflib.SequenceMatcher(a=tok, b=k).ratio() for k in keys), default=0.0)
            if best >= 0.78:  # próg tolerancji
                return mood
    return None


def _spotify_search_url(q: str) -> str:
    return f"https://open.spotify.com/search/{_url.quote(q)}"

def _youtube_search_url(q: str) -> str:
    return f"https://www.youtube.com/results?search_query={_url.quote(q)}"

def pick_gtzan_clip(genre: str, base_dir: str = "data/gtzan"):
    exts = ("*.wav","*.au","*.mp3")
    files = []
    for ext in exts:
        files += glob.glob(os.path.join(base_dir, genre, ext))
    return random.choice(files) if files else None

# --- VIBE FIRST (jedna wersja) ---
st.subheader("🎛️ Vibe first")
colA, colB = st.columns([2, 3])
with colA:
    preset = st.selectbox(
        "Wybierz nastrój",
        ["(wybierz)…"] + list(MOOD_TO_GTZAN.keys()),
        index=0
    )
with colB:
    free = st.text_input(
        "…albo wpisz własnymi słowami",
        placeholder="Np. 'potrzebuję energii ⚡ do treningu'"
    )

apply_clicked = st.button("Zastosuj vibe", key="apply_vibe")

# ustal mood: preset > tekst (po kliknięciu lub Enter)
mood = None
if preset != "(wybierz)…":
    mood = preset
elif apply_clicked or free:
    mood = _mood_from_text(free)

if mood:
    apply_mood_background(mood)
    st.caption(f"🎨 Tło dopasowane do nastroju: **{mood}**")
    render_equalizer()

    genre = MOOD_TO_GTZAN.get(mood, "pop")
    demo_path = pick_gtzan_clip(genre, base_dir=os.getenv("GTZAN_PATH", "data/gtzan"))
    q = f"{mood} {genre} playlist"

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("🎵 Gatunek (demo)", genre)
    with c2: st.metric("🧠 Mood", mood)
    with c3: st.metric("🔎 Zapytanie", q)

    st.write("**Proponowany odsłuch:**")
    cA, cB, cC = st.columns([2, 2, 3])
    with cA: st.link_button("▶️ Spotify – szukaj", _spotify_search_url(q))
    with cB: st.link_button("▶️ YouTube – szukaj", _youtube_search_url(q))
    with cC:
        if demo_path:
            st.success("🎧 Lokalny klip demo (GTZAN)")
            st.audio(demo_path)
        else:
            st.info("Brak lokalnego klipu GTZAN dla tego gatunku (ustaw GTZAN_PATH lub wgraj pliki).")
else:
    if apply_clicked or free:
        st.info("Nie rozpoznałem nastroju — wybierz z listy po lewej lub spróbuj innymi słowami (np. 'energia', 'chill', 'fokus').")

st.divider()


audio_file = st.file_uploader("Wrzuć plik audio", type=["wav", "mp3", "ogg"])

# --- Model info ---
try:
    model = load_model()
    if model is None:
        st.warning("Nie znaleziono modelu **models/baseline_svc.joblib**. "
                   "Uruchom trening: `python src/scripts/train_baseline.py`.")
    else:
        name = type(getattr(model, 'steps', [[model]])[-1][-1]).__name__ if hasattr(model, 'steps') else type(model).__name__
        st.caption(f"✅ Model załadowany: **{name}**")
except Exception as e:
    st.error(f"Problem z załadowaniem modelu: {e}")

# --- Inference ---
if audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{audio_file.name}") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name
    try:
        pred_class, conf, classes, probs = predict_genre(tmp_path)
        mood = map_genre_to_mood(pred_class)

        apply_mood_background(mood)

        st.markdown(f"## 🎵 Gatunek: **{pred_class}** · pewność **{conf:.2f}**")
        st.markdown(f"**Klimat:** `{mood}`")
        render_equalizer()
        st.audio(tmp_path)

        if isinstance(probs, (list, tuple, np.ndarray)) and len(probs) > 1:
            order = np.argsort(probs)[::-1][:5]
            st.write("Top-5:")
            for i in order:
                st.write(f"- {classes[i]}: {probs[i]:.2f}")
    except Exception as e:
        # FIX: to musi być wcięte pod except
        st.error(f"Nie udało się zanalizować audio: {e}")
else:
    st.info("Wybierz plik audio (10–30 s działa najlepiej).")

    # --- Feedback box (pojawia się, gdy nie wgrano pliku) ---
    with st.expander("💬 Zostaw pytanie / feedback"):
        # blokada wielokrotnego wysyłania
        if "fb_sent" not in st.session_state:
            st.session_state.fb_sent = False

        uname = st.text_input("Twoje imię (opcjonalnie)", key="fb_name")
        umail = st.text_input("Email (opcjonalnie)", key="fb_mail")
        umsg  = st.text_area(
            "Twoja wiadomość",
            key="fb_msg",
            placeholder="Napisz, co poprawić / co działa super…",
            help="Minimum 5 znaków, unikaj wrażliwych danych."
        )

        disabled = st.session_state.fb_sent
        if st.button("Wyślij", key="fb_send", disabled=disabled):
            msg  = (umsg or "").strip()
            name = (uname or "").strip()
            mail = (umail or "").strip()

            # walidacje
            if len(msg) < 5:
                st.warning("Dodaj trochę więcej treści (min. 5 znaków).")
                st.stop()
            if not is_valid_email(mail):
                st.warning("Podaj poprawny adres email (albo zostaw puste).")
                st.stop()

            row = {
                "id": str(uuid.uuid4()),
                "ts": datetime.now().isoformat(timespec="seconds"),
                "name": name,
                "email": mail,
                "message": msg.replace("\r", " ").strip(),
                # meta (opcjonalnie z session_state)
                "app_version": st.session_state.get("app_version", "v1.5"),
                "model": st.session_state.get("model_name", ""),
                "session_id": st.session_state.get("session_id", ""),
            }

            try:
                save_feedback(row)
                st.session_state.fb_sent = True
                st.success("Dzięki! Wiadomość zapisana ✅")
                # wyczyść pola
                st.session_state.fb_name = ""
                st.session_state.fb_mail = ""
                st.session_state.fb_msg = ""
            except Exception as e:
                st.error(f"Nie udało się zapisać feedbacku: {e}")
                st.session_state.fb_sent = False
