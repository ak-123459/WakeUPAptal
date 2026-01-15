
#!/usr/bin/env python3
"""
Data Collection Pipeline
------------------------
• Downloads and extracts LibriSpeech (speech) and ESC-50 (noise) datasets
• Processes audio into 5s WAV clips at 16kHz
• Handles mono conversion, normalization, and resampling
• CLI-driven, dynamic, and ML-ready
"""

import os
import random
import tarfile
import zipfile
import shutil
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm
import urllib.request
import requests
import librosa

# -----------------------------
# DEFAULT CONFIGS
# -----------------------------
LIBRISPEECH = {
    "name": "LibriSpeech/train-clean-100",
    "url": "https://openslr.trmal.net/resources/12/train-clean-100.tar.gz",
    "format": "tar.gz"
}

ESC50 = {
    "name": "ESC-50",
    "url": "https://github.com/karolpiczak/ESC-50/archive/master.zip",
    "format": "zip"
}


# =============================
# UTILITIES: DOWNLOAD & EXTRACT
# =============================
def download_file(url: str, output_path: Path):
    if output_path.exists():
        return True
    print(f"⬇️ Downloading {output_path.name}...")
    if url.endswith(".zip") or url.endswith(".tar.gz"):
        urllib.request.urlretrieve(url, output_path)
    else:
        # requests stream download
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with open(output_path, "wb") as f, tqdm(
            desc=f"Downloading {output_path.name}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    print(f"✅ Download complete: {output_path.name}")
    return True


def extract_tar(tar_path: Path, extract_to: Path):
    if extract_to.exists():
        print(f"✅ Already extracted: {extract_to}")
        return
    print(f"📦 Extracting {tar_path.name}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tqdm(tar.getmembers(), desc="Extracting", unit="files"):
            tar.extract(member, path=extract_to.parent)
    print(f"✅ Extraction done: {extract_to}")


def extract_zip(zip_path: Path, extract_to: Path):
    if extract_to.exists():
        print(f"✅ Already extracted: {extract_to}")
        return
    print(f"📦 Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Extraction done: {extract_to}")


# =============================
# UTILITIES: AUDIO PROCESSING
# =============================
def process_audio_file(
    input_path: Path,
    output_dir: Path,
    clip_duration: float = 5.0,
    target_sr: int = 16000,
    start_offset: float = 1.0,
):
    """Convert audio file to WAV clip(s)"""
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / input_path.with_suffix(".wav").name
    if wav_path.exists():
        return [wav_path]

    try:
        audio, sr = sf.read(input_path, dtype="float32") if input_path.suffix == ".flac" else librosa.load(input_path, sr=target_sr, mono=True)
        # Convert to mono if needed
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)
        # Resample if needed
        if sr != target_sr:
            audio = librosa.resample(audio, sr, target_sr)
        # Clip
        start = int(start_offset * target_sr)
        end = start + int(clip_duration * target_sr)
        audio = audio[start:end] if len(audio) >= end else audio[-int(clip_duration * target_sr):]
        # Normalize
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.95
        sf.write(wav_path, audio, target_sr, subtype="PCM_16")
        return [wav_path]
    except Exception as e:
        print(f"⚠️ Error processing {input_path.name}: {e}")
        return []


def process_folder(input_folder: Path, output_folder: Path, clip_duration=5.0, target_sr=16000):
    audio_exts = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"]
    files = [f for f in input_folder.rglob("*") if f.suffix.lower() in audio_exts]
    print(f"🎧 Found {len(files)} audio files in {input_folder}")
    all_outputs = []
    for f in tqdm(files, desc="Processing audio"):
        all_outputs.extend(process_audio_file(f, output_folder, clip_duration, target_sr))
    print(f"✅ Created {len(all_outputs)} clips in {output_folder}")
    return all_outputs


# =============================
# LIBRISPEECH PIPELINE
# =============================
def prepare_librispeech(project_root: Path, output_dir: Path, max_files: int = 10000):
    data_root = project_root / "data"
    src_root = data_root / LIBRISPEECH["name"]
    out_dir = project_root / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tar_path = data_root / f"{LIBRISPEECH['name'].split('/')[-1]}.tar.gz"
    data_root.mkdir(exist_ok=True)

    download_file(LIBRISPEECH["url"], tar_path)
    extract_tar(tar_path, src_root)

    # Collect FLAC files
    flac_files = list(src_root.rglob("*.flac"))
    print(f"🎧 Found {len(flac_files)} FLAC files")
    random.shuffle(flac_files)
    selected_files = flac_files[:min(max_files, len(flac_files))]

    saved = 0
    for f in tqdm(selected_files, desc="Processing LibriSpeech"):
        saved += len(process_audio_file(f, out_dir))
    print(f"✅ LibriSpeech done: {saved} WAV files → {out_dir}")


# =============================
# ESC-50 PIPELINE
# =============================
def prepare_esc50(project_root: Path, output_dir: Path):
    print("\n=== Preparing ESC-50 Noise Dataset ===\n")
    temp_dir = project_root / "temp_esc50"
    temp_dir.mkdir(exist_ok=True)
    zip_path = temp_dir / "esc50.zip"
    download_file(ESC50["url"], zip_path)
    extract_zip(zip_path, temp_dir)

    # Locate audio folder
    audio_folder = None
    for root, dirs, _ in os.walk(temp_dir):
        if "audio" in dirs:
            audio_folder = Path(root) / "audio"
            break
    if not audio_folder:
        print("⚠️ Could not find audio folder in ESC-50")
        return

    out_dir = project_root / output_dir
    process_folder(audio_folder, out_dir)
    shutil.rmtree(temp_dir)
    print(f"✅ ESC-50 noise dataset ready: {out_dir}")


# =============================
# MAIN
# =============================
def main():
    parser = argparse.ArgumentParser(description="Speech + Noise Data Collection Pipeline")
    parser.add_argument("--project-root", default=".", help="Project root")
    parser.add_argument("--speech-output", default="speeches", help="Output folder for speech")
    parser.add_argument("--noise-output", default="noises", help="Output folder for noise")
    parser.add_argument("--max-speech-files", type=int, default=10000)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()

    prepare_librispeech(project_root, args.speech_output, args.max_speech_files)
    prepare_esc50(project_root, args.noise_output)


if __name__ == "__main__":
    main()
