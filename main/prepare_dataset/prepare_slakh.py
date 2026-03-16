"""
Download and prepare Slakh2100 at 44100 Hz.

Downloads the original Slakh2100 dataset from Zenodo, groups stems by
instrument class (bass, drums, guitar, piano), mixes stems of the same
class into a single mono WAV, creates mixture.wav, saves split metadata
JSON, and removes the raw download.

Output structure:
    dataset/slakh2100_44100/
        train/Track00001/{bass,drums,guitar,piano,mixture}.wav
        validation/...
        test/...
        metadata/{train,validation,test}.json

Sources:
    Full dataset  (~100 GB, 44100 Hz): Zenodo 4599666
    Tiny subset   (~ 880 MB,  16 kHz): Zenodo 4603870  --tiny flag, for testing only

Both archives are flat (all TrackXXXXX dirs at the root, no pre-made splits).
Tracks are assigned to splits by track number following the official convention:
    train:      Track00001 – Track01500
    validation: Track01501 – Track01875
    test:       Track01876 – Track02100

Usage:
    # Quick test (~880 MB download, 16 kHz)
    python main/prepare_dataset/prepare_slakh.py --splits test --tiny

    # Full dataset (44100 Hz, ~100 GB)
    python main/prepare_dataset/prepare_slakh.py --splits train validation test

    # Already downloaded — skip download
    python main/prepare_dataset/prepare_slakh.py --raw_dir /path/to/slakh2100 --splits train validation test
"""

import argparse
import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import torchaudio
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Zenodo sources
# ---------------------------------------------------------------------------
ZENODO_URL      = "https://zenodo.org/api/records/4599666/files/slakh2100_flac_redux.tar.gz/content"  # ~100 GB, 44100 Hz
ZENODO_URL_TINY = "https://zenodo.org/api/records/4603870/files/babyslakh_16k.tar.gz/content"         # ~880 MB, 16 kHz (testing only)

TARGET_SR = 44100
STEMS = ["bass", "drums", "guitar", "piano"]

# Official Slakh2100 split boundaries (by track number)
SPLIT_RANGES = {
    "train":      range(1, 1501),
    "validation": range(1501, 1876),
    "test":       range(1876, 2101),
}

# inst_class values in metadata.yaml that map to our 4 stems
INST_CLASS_MAP = {
    "Bass":   "bass",
    "Guitar": "guitar",
    "Piano":  "piano",
    "Drums":  "drums",
}


def track_num(track_name: str) -> int:
    return int(track_name.replace("Track", ""))


def assign_split(track_name: str) -> str:
    n = track_num(track_name)
    for split, rng in SPLIT_RANGES.items():
        if n in rng:
            return split
    return "train"


def load_and_mix_stems(stem_paths: list, target_sr: int) -> torch.Tensor:
    mixed = None
    for p in stem_paths:
        wav, sr = torchaudio.load(str(p))
        if sr != target_sr:
            wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        wav = wav.mean(dim=0, keepdim=True)  # → mono
        mixed = wav if mixed is None else mixed + wav
    return mixed


def process_track(track_dir: Path, out_dir: Path, target_sr: int) -> dict | None:
    yaml_path = track_dir / "metadata.yaml"
    if not yaml_path.exists():
        return None

    yaml_text = yaml_path.read_text()
    if yaml_text and yaml_text[0] not in ["{", "-", " ", "\n"]:
        yaml_text = yaml_text[1:]
    metadata = yaml.safe_load(yaml_text)

    stems_dir = track_dir / "stems"
    if not stems_dir.exists():
        return None

    # Group stems by our 4 classes using inst_class field
    class_stems: dict[str, list] = {c: [] for c in STEMS}
    for stem_id, stem_meta in metadata.get("stems", {}).items():
        inst_class = stem_meta.get("inst_class", "")
        cls = INST_CLASS_MAP.get(inst_class)
        if cls is None:
            continue
        # Support both .wav and .flac
        for ext in [".wav", ".flac"]:
            f = stems_dir / f"{stem_id}{ext}"
            if f.exists():
                class_stems[cls].append(f)
                break

    if all(len(v) == 0 for v in class_stems.values()):
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    # Find reference length from any available stem
    ref_audio = None
    for cls in STEMS:
        if class_stems[cls]:
            ref_audio = load_and_mix_stems(class_stems[cls][:1], target_sr)
            break

    mixed_stems = {}
    for cls in STEMS:
        if class_stems[cls]:
            audio = load_and_mix_stems(class_stems[cls], target_sr)
        else:
            audio = torch.zeros(1, ref_audio.shape[-1])
        torchaudio.save(str(out_dir / f"{cls}.wav"), audio, target_sr)
        mixed_stems[cls] = audio

    # mixture
    max_len = max(a.shape[-1] for a in mixed_stems.values())
    mixture = sum(
        torch.nn.functional.pad(a, (0, max_len - a.shape[-1]))
        for a in mixed_stems.values()
    )
    torchaudio.save(str(out_dir / "mixture.wav"), mixture, target_sr)

    return {
        track_dir.name: {
            "length": int(mixture.shape[-1]),
            "mean": float(mixture.mean()),
            "std": float(mixture.std()),
            "samplerate": target_sr,
        }
    }


def prepare_split(tracks: list, out_split_dir: Path, target_sr: int, num_workers: int) -> dict:
    metadata = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_track, t, out_split_dir / t.name, target_sr): t
            for t in tracks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=out_split_dir.name):
            result = future.result()
            if result:
                metadata.update(result)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=str, default="dataset/slakh2100_44100",
                        help="Output directory")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"],
                        choices=["train", "validation", "test"])
    parser.add_argument("--max_tracks", type=int, default=None,
                        help="Limit tracks per split (for quick testing)")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--keep_raw", action="store_true",
                        help="Do not delete the raw download after processing")
    parser.add_argument("--raw_dir", type=str, default=None,
                        help="Path to already-extracted Slakh2100 directory (skips download)")
    parser.add_argument("--tiny", action="store_true",
                        help="Download BabySlakh (~880 MB, 16 kHz) instead of full dataset. For testing only.")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    # ── Download if needed ──────────────────────────────────────────────────
    if args.raw_dir:
        raw_root = Path(args.raw_dir)
        tmp_dir = None
    else:
        tmp_dir = tempfile.mkdtemp()
        raw_root = Path(tmp_dir)
        url = ZENODO_URL_TINY if args.tiny else ZENODO_URL
        filename = "babyslakh_16k.tar.gz" if args.tiny else "slakh2100_flac_redux.tar.gz"
        print(f"Downloading {'BabySlakh (testing)' if args.tiny else 'Slakh2100 (~100 GB)'} ...")
        if not args.tiny:
            print("WARNING: this will download ~100 GB. Use --tiny for a quick test.")
        from torchvision.datasets.utils import download_and_extract_archive
        download_and_extract_archive(url, raw_root, filename=filename, remove_finished=not args.keep_raw)

    # ── Find extracted root ─────────────────────────────────────────────────
    # After download the archive extracts into a subdir (e.g. babyslakh_16k/).
    # If --raw_dir was given, use it directly.
    if args.raw_dir:
        slakh_root = Path(args.raw_dir)
    else:
        candidates = [d for d in raw_root.iterdir() if d.is_dir()]
        slakh_root = candidates[0] if candidates else raw_root

    # ── Process each requested split ────────────────────────────────────────
    # Both full and BabySlakh are flat (TrackXXXXX dirs at root).
    # Splits are assigned by track number following the official convention.
    all_tracks = sorted([d for d in slakh_root.iterdir() if d.is_dir() and d.name.startswith("Track")])

    meta_dir = dest / "metadata"
    meta_dir.mkdir(exist_ok=True)

    for split in args.splits:
        tracks = [t for t in all_tracks if assign_split(t.name) == split]

        if args.max_tracks:
            tracks = tracks[:args.max_tracks]

        if not tracks:
            print(f"No tracks found for split '{split}', skipping.")
            continue

        print(f"\nProcessing {split}: {len(tracks)} tracks ...")
        split_meta = prepare_split(tracks, dest / split, TARGET_SR, args.num_workers)

        meta_path = meta_dir / f"{split}.json"
        with open(meta_path, "w") as f:
            json.dump(split_meta, f, indent=4)
        print(f"Saved metadata → {meta_path}  ({len(split_meta)} tracks)")

    # ── Clean up ────────────────────────────────────────────────────────────
    if tmp_dir and not args.keep_raw:
        print("Removing raw download ...")
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\nDone. Dataset at: {dest}")


if __name__ == "__main__":
    main()
