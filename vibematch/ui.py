# vibematch/ui.py
import streamlit as st

# bazowe kolory dark (zostawiamy jak masz)
DARK_BG   = "#0f1216"
DARK_TEXT = "#e8edf2"
CARD_BG   = "#151a20"
BORDER    = "#2a3443"

def apply_dark_base():
    st.markdown(f"""
    <style>
      .stApp {{ background: {DARK_BG}; color: {DARK_TEXT}; }}
      h1,h2,h3,h4,h5,h6, p, span, label {{ color: {DARK_TEXT} !important; }}
      .stAlert > div {{ background:{CARD_BG}; color:{DARK_TEXT}; border:1px solid {BORDER}; }}
      .stFileUploader > div {{ background:{CARD_BG}; border:1px solid {BORDER}; }}
      .stTextInput > div > div, .stTextArea > div > div {{ background:{CARD_BG}; border:1px solid {BORDER}; }}
      .stButton > button {{ background:{CARD_BG}; color:{DARK_TEXT}; border:1px solid {BORDER}; }}
    </style>
    """, unsafe_allow_html=True)

# ➜ NOWA WYRAŹNA PALETA (klucze zgodne z app.py: sunny/focus/romance/chill/energy/happy/party/zen/melancholy/confidence)
MOOD_GRADIENTS = {
    "sunny":       "linear-gradient(120deg, #FFB703 0%, #FB8500 45%, #FFD166 100%)",  # pomarańcze/słońce
    "focus":       "linear-gradient(160deg, #0B1220 0%, #14532D 60%, #22C55E 100%)",  # stonowana zieleń
    "romance":     "linear-gradient(135deg, #7F1D1D 0%, #DC2626 50%, #F87171 100%)",  # czerwienie
    "chill":       "linear-gradient(160deg, #0E7490 0%, #38BDF8 50%, #A5F3FC 100%)",  # turkus/lofi
    "energy":      "linear-gradient(135deg, #7C2D12 0%, #DC2626 45%, #F59E0B 100%)",  # ogień
    "happy":       "linear-gradient(135deg, #22C55E 0%, #10B981 45%, #FBBF24 100%)",  # radośnie
    "party":       "linear-gradient(135deg, #6D28D9 0%, #A21CAF 45%, #EF4444 100%)",  # klubowo
    "zen":         "linear-gradient(160deg, #0B1220 0%, #334155 50%, #64748B 100%)",  # przygaszone szaro-niebieskie
    "melancholy":  "linear-gradient(160deg, #111827 0%, #1F2937 50%, #374151 100%)",  # ciemne szarości
    "confidence":  "linear-gradient(135deg, #0B1220 0%, #0EA5E9 45%, #22D3EE 100%)",  # chłodno-mocno
    # opcjonalne alternatywy:
    "romance_alt": "linear-gradient(135deg, #581C87 0%, #BE185D 50%, #F472B6 100%)",
    "focus_alt":   "linear-gradient(160deg, #052e16 0%, #166534 60%, #4ade80 100%)",
    "sunny_alt":   "linear-gradient(120deg, #f59e0b 0%, #f97316 50%, #fde68a 100%)",
}

def apply_mood_background(mood: str) -> None:
    key = (mood or "chill").lower()
    g = MOOD_GRADIENTS.get(key, MOOD_GRADIENTS["chill"])
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {g} !important;
            background-attachment: fixed !important;
            transition: background 0.35s ease;
        }}
        /* lekka mgiełka pod treść dla czytelności na jaskrawych tłach */
        .block-container {{
            background: rgba(0,0,0,0.12);
            border-radius: 16px;
            padding: 1rem 1.25rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_equalizer():
    st.markdown("""
    <style>
    .eq{display:inline-flex;gap:6px;height:28px;align-items:flex-end;margin:.5rem 0 1rem}
    .bar{width:6px;background:#e8edf2;opacity:.9;animation:bump 1s infinite;border-radius:3px}
    .bar:nth-child(1){height:18px;animation-delay:0s}
    .bar:nth-child(2){height:12px;animation-delay:.1s}
    .bar:nth-child(3){height:22px;animation-delay:.2s}
    .bar:nth-child(4){height:10px;animation-delay:.3s}
    .bar:nth-child(5){height:20px;animation-delay:.4s}
    @keyframes bump{0%,100%{transform:scaleY(.6)}50%{transform:scaleY(1.25)}}
    </style>
    <div class="eq"><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div></div>
    """, unsafe_allow_html=True)
