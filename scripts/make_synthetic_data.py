from pathlib import Path
import numpy as np
import soundfile as sf

SR = 22050
DUR = 2.0  # sekundy
N = int(SR * DUR)

OUT = Path("data/raw")
CLASSES = {
    "sine": 440.0,      # A4 czysty sinus
    "square": 220.0,    # fala kwadratowa (zgrubnie)
}

def square_wave(freq, t):
    return np.sign(np.sin(2*np.pi*freq*t))

OUT.mkdir(parents=True, exist_ok=True)

for name, freq in CLASSES.items():
    cls_dir = OUT / name
    cls_dir.mkdir(parents=True, exist_ok=True)
    for i in range(8):  # po 8 próbek na klasę
        t = np.linspace(0, DUR, N, endpoint=False)
        if name == "sine":
            y = np.sin(2*np.pi*freq*t)
        else:
            y = square_wave(freq, t)
        # lekkie szumy, różne amplitudy
        y = 0.5 * y + 0.05 * np.random.randn(N)
        sf.write(cls_dir / f"{name}_{i:02d}.wav", y, SR)

print("OK: synthetic data generated in data/raw/{sine,square}")
