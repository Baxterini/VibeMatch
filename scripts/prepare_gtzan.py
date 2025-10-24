from pathlib import Path
import argparse, random
import librosa, soundfile as sf

GENRES_10 = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
EXTS = ("*.wav","*.au","*.mp3")

def collect_files(src: Path):
    by_genre = {}
    for d in sorted([p for p in src.iterdir() if p.is_dir()]):
        files = []
        for ext in EXTS:
            files.extend(d.glob(ext))
        if files:
            by_genre[d.name.lower()] = sorted(files)
    return by_genre

def ensure_wav_mono_22050(in_path: Path, out_path: Path, sr=22050):
    y, _sr = librosa.load(in_path, sr=sr, mono=True)
    sf.write(out_path, y, sr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source","-s", required=True, help="Folder z podkatalogami gatunków (blues, jazz, ...)")
    ap.add_argument("--per-class","-k", type=int, default=20, help="Ile plików na gatunek")
    ap.add_argument("--out","-o", default="data/raw", help="Folder wyjściowy")
    args = ap.parse_args()

    src = Path(args.source).expanduser().resolve()
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)

    by_genre = collect_files(src)
    if not by_genre:
        raise SystemExit(f"Brak plików audio w {src}")

    genres = [g for g in GENRES_10 if g in by_genre] or list(by_genre.keys())[:10]
    print("Gatunki:", genres)

    for g in genres:
        files = by_genre[g]
        pick = files if len(files) <= args.per_class else random.sample(files, args.per_class)
        out_dir = out_root / g; out_dir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(pick):
            ensure_wav_mono_22050(f, out_dir / f"{g}_{i:03d}.wav")
        print(f"[{g}] zapisano {len(pick)} plików → {out_dir}")

    print("✔ Mini-GTZAN gotowy w", out_root)

if __name__ == "__main__":
    main()
