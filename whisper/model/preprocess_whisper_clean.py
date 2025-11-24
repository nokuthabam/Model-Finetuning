import os
import warnings
import logging
import torchaudio
import numpy as np
from pathlib import Path
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import WhisperProcessor

# -----------------------------------------
# SUPPRESS WARNINGS
# -----------------------------------------
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torchaudio").setLevel(logging.ERROR)

# -----------------------------------------
# CONFIG
# -----------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAVE_DIR = BASE_DIR / "model" / "processed_arrow"
MODEL_NAME = "openai/whisper-small"

LANGS = {
    "siswati": {
        "json_files": [
            DATA_DIR / "siswati_dataset.jsonl",
            DATA_DIR / "nchlt_ss_whisper.jsonl",
        ]
    },
}

processor = WhisperProcessor.from_pretrained(MODEL_NAME)

# -----------------------------------------
# NORMALIZE TEXT COLUMN
# -----------------------------------------
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

    drop = [
        "label", "client_id", "path", "variant", "segment",
        "gender", "accent", "locale", "waveform", "duration"
    ]
    for c in drop:
        if c in ds.column_names:
            ds = ds.remove_columns(c)

    return ds

# -----------------------------------------
# LOAD JSON DATA
# -----------------------------------------
def load_json_data(json_paths, max_samples=999999):
    datasets = []

    for json_file in json_paths:
        if not json_file.exists():
            print(f"⚠️ Skipping missing: {json_file}")
            continue

        print(f"📥 Loading {json_file.name}")
        ds = load_dataset("json", data_files=str(json_file), split="train")

        if "audio_path" in ds.column_names and "audio" not in ds.column_names:
            print("🔁 Renaming audio_path → audio")
            ds = ds.rename_column("audio_path", "audio")

        ds = ds.cast_column("audio", Audio(sampling_rate=16000, decode=False))

        if len(ds) > max_samples:
            ds = ds.select(range(max_samples))

        ds = normalize_text_columns(ds)
        if ds is not None:
            datasets.append(ds)

    return concatenate_datasets(datasets) if datasets else None

# -----------------------------------------
# DURATION
# -----------------------------------------
def compute_duration_lazy(sample):
    ap = sample["audio"]["path"]
    info = torchaudio.info(ap)
    sample["duration"] = info.num_frames / info.sample_rate
    return sample

def filter_duration(sample):
    return 0 < sample["duration"] <= 30.0

# -----------------------------------------
# PREPARE BATCH
# -----------------------------------------
def prepare_batch(batch):
    audio_path = batch["audio"]["path"]

    audio, sr = torchaudio.load(audio_path)
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)

    samples = audio.squeeze().numpy()

    mel = processor.feature_extractor(samples, sampling_rate=16000).input_features[0]
    text = batch["text"]
    label_ids = processor.tokenizer(text).input_ids

    batch["input_features"] = np.array(mel, dtype=np.float32)
    batch["labels"] = np.array(label_ids, dtype=np.int64)
    return batch

# -----------------------------------------
# MAIN
# -----------------------------------------
if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    for lang, cfg in LANGS.items():
        print(f"\n===============================")
        print(f"🌍 LANGUAGE: {lang}")
        print("===============================")

        ds = load_json_data(cfg["json_files"])
        if ds is None:
            print(f"❌ No dataset for {lang}")
            continue

        print(f"📦 Loaded dataset: {len(ds)} samples")

        ds = ds.map(compute_duration_lazy, num_proc=32)
        ds = ds.filter(filter_duration)

        keep_cols = ["text"]
        remove_cols = [c for c in ds.column_names if c not in keep_cols]

        print("🎧 Extracting Whisper features…")
        ds = ds.map(
            prepare_batch,
            remove_columns=remove_cols,
            num_proc=1,
            batched=False,
        )

        out_dir = SAVE_DIR / lang
        print(f"💾 Saving Arrow dataset → {out_dir}")
        ds.save_to_disk(str(out_dir), num_proc=8)

        print(f"✅ DONE: saved {lang}")
