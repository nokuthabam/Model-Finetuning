
# preprocess_whisper_clean.py
import os
from pathlib import Path
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import WhisperProcessor
import numpy as np
import torchaudio

# Suppress warnings from torchaudio
import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Tensorflow
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torchaudio").setLevel(logging.ERROR)



# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAVE_DIR = BASE_DIR / "model" / "processed_arrow"
MODEL_NAME = "openai/whisper-small"

LANGS = {
    # "siswati": {
    #     "json_files": [
    #         DATA_DIR / "siswati_dataset.jsonl",
    #         DATA_DIR / "nchlt_ss_whisper.jsonl",
    #     ]
    # },
    "zulu": {
        "json_files": [
            DATA_DIR / "zulu_dataset.jsonl",
            DATA_DIR / "nchlt_zu_whisper.jsonl",
            DATA_DIR / "commonvoice_zu_train.json",
        ]
    },
    "xhosa": {
        "json_files": [
            DATA_DIR / "xhosa_dataset.jsonl",
            DATA_DIR / "nchlt_xh_whisper.jsonl",
            DATA_DIR / "commonvoice_xh_train.json",
        ]
    },
    "ndebele": {
        "json_files": [
            DATA_DIR / "ndebele_dataset.jsonl",
            DATA_DIR / "nchlt_nr_whisper.jsonl",
        ]
    },
}

processor = WhisperProcessor.from_pretrained(MODEL_NAME)

# -------------------------------------------------------------------
# TEXT NORMALIZATION
# -------------------------------------------------------------------
def normalize_text_columns(ds):
    cols = ds.column_names
    if "transcription" in cols:
        ds = ds.rename_column("transcription", "text")
    elif "sentence" in cols:
        ds = ds.rename_column("sentence", "text")
    elif "transcript" in cols:
        ds = ds.rename_column("transcript", "text")
    elif "text" not in cols:
        print(f"⚠️ No text column found. Columns: {cols}")
        return None

    # Remove irrelevant metadata
    remove_cols = [
        "label", "client_id", "path", "variant", "segment",
        "gender", "accent", "locale", "duration", "waveform"
    ]
    for c in remove_cols:
        if c in ds.column_names:
            ds = ds.remove_columns(c)

    return ds

# -------------------------------------------------------------------
# LOAD + CLEAN
# -------------------------------------------------------------------
def load_json_data(json_paths, max_samples=1e6):
    datasets = []

    for json_file in json_paths:
        if not json_file.exists():
            print(f"⚠️ Skipping missing file: {json_file}")
            continue

        print(f"📥 Loading {json_file.name}")
        ds = load_dataset("json", data_files=str(json_file), split="train")

        print("🔎 First sample before processing:", ds[0])

        # 💡 Only rename if audio_path exists
        if "audio_path" in ds.column_names and "audio" not in ds.column_names:
            print("🔁 Renaming 'audio_path' → 'audio'")
            ds = ds.rename_column("audio_path", "audio")

        # 🧠 Use lazy loading (no decoding) to avoid RAM issues
        print("🎧 Casting audio column with decode=False")
        ds = ds.cast_column("audio", Audio(sampling_rate=16000, decode=False))

        # ✂️ Optional: sample a small subset for quick testing
        if len(ds) > max_samples:
            print(f"⚠️ Sampling only {max_samples} out of {len(ds)} entries to avoid OOM")
            ds = ds.select(range(max_samples))

        # 📝 Normalize transcript column
        ds = normalize_text_columns(ds)
        datasets.append(ds)

    return concatenate_datasets(datasets) if datasets else None


# -------------------------------------------------------------------
# DURATION + VALIDITY FILTERING
# -------------------------------------------------------------------
def compute_duration_lazy(sample):
    audio_path = sample["audio"]["path"]
    info = torchaudio.info(audio_path)
    sample["duration"] = info.num_frames / info.sample_rate
    return sample


def valid_audio(sample):
    audio = sample.get("audio", {})
    return audio is not None and audio.get("array") is not None

def filter_by_duration(sample):
    return 0 < sample["duration"] <= 30.0

# -------------------------------------------------------------------
# PREPARE FOR WHISPER
# -------------------------------------------------------------------
def prepare_batch(batch):
    audio_info = batch["audio"]
    audio_path = audio_info["path"] if isinstance(audio_info, dict) else audio_info

    # 👇 Decode audio from file path using torchaudio
    array, sr = torchaudio.load(audio_path)

    if sr != 16000:
        array = torchaudio.functional.resample(array, orig_freq=sr, new_freq=16000)

    batch["input_features"] = processor.feature_extractor(array.squeeze().numpy(), sampling_rate=16000).input_features[0]

    transcription = batch.get("transcription", batch.get("text", batch.get("transcript", "")))
    batch["labels"] = processor.tokenizer(transcription).input_ids
    return batch

# -------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    for lang, cfg in LANGS.items():
        print(f"\n===============================")
        print(f"🌍 LANGUAGE: {lang}")
        print("===============================")

        ds = load_json_data(cfg["json_files"])
        print(f"📦 Initial dataset size: {len(ds) if ds else 0}")
        #Select a subset for testing
        # ds = ds.select(range(min(len(ds), 10)))
        if ds is None:
            print(f"❌ No dataset for {lang}")
            continue
        
        print(f"⏱️ Computing durations...")
        ds = ds.map(compute_duration_lazy, num_proc=32)
        valid = ds.filter(lambda s: s["duration"] > 0)
        print(f"✅ Valid: {len(valid)} / {len(ds)}")
        print(valid["duration"][:10])  # First 10 durations


        print(f"🔍 Filtering samples > 30s...")
        ds = ds.filter(filter_by_duration)
        print(f"📦 Remaining after filtering: {len(ds)}")

        print(f"🎧 Computing Whisper mel + labels…")
        keep_cols = ["audio", "text"]
        remove_cols = [c for c in ds.column_names if c not in keep_cols]

        ds = ds.map(
            prepare_batch,
            remove_columns=remove_cols,
            num_proc=1,
            batched=False,
            desc=f"Processing {lang}",
        )

        lang_dir = SAVE_DIR / lang
        os.makedirs(lang_dir, exist_ok=True)
        ds.save_to_disk(str(lang_dir))

        print(f"✅ Saved Arrow dataset → {lang_dir}")
