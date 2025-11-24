# e:/Model-Finetuning/whisper/model/preprocess.py
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import WhisperProcessor
from pathlib import Path
from pydub import AudioSegment, silence
import numpy as np
import re
import unicodedata
import os
import warnings
import logging


# -----------------------------------------
# SUPPRESS WARNINGS
# -----------------------------------------
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
    # "ndebele": {"json_files": [BASE_DIR / "merged_nr.json"]},
    # "siswati": {"json_files": [BASE_DIR / "merged_ss.json"]},
    "zulu": {"json_files": [BASE_DIR / "merged_zu.json"]},
    "xhosa": {"json_files": [BASE_DIR / "merged_xh.json"]},
}


processor = WhisperProcessor.from_pretrained(MODEL_NAME)


# -------------------------------------------------------------------
# 🔤 TEXT CLEANING PIPELINE
# -------------------------------------------------------------------

def clean_text(text: str):
    if not isinstance(text, str):
        return ""

    # 1. remove [brackets] but KEEP inner content
    text = re.sub(r"\[([^]]+)\]", r"\1", text)

    # 2. remove any stray brackets left
    text = text.replace("[", "").replace("]", "")

    # 3. normalize unicode accents and weird spacing
    text = unicodedata.normalize("NFKC", text)

    # 4. collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # 5. lowercase (Whisper small prefers lowercase training)
    return text.lower()


# -------------------------------------------------------------------
# 🔇 SILENCE TRIMMING FOR EACH AUDIO FILE
# -------------------------------------------------------------------

def trim_audio(audio_array, sampling_rate=16000, top_db=35):
    """
    Trims leading/trailing silence using pydub silence detection.
    """
    try:
        sound = AudioSegment(
            audio_array.tobytes(),
            frame_rate=sampling_rate,
            sample_width=audio_array.dtype.itemsize,
            channels=1
        )

        # detect silence (threshold = 30–40 dB usually best)
        chunks = silence.split_on_silence(
            sound,
            min_silence_len=300,      # 0.3 sec
            silence_thresh=sound.dBFS - top_db,
            keep_silence=150
        )

        if len(chunks) == 0:
            return audio_array  # nothing trimmed

        trimmed = chunks[0]
        for c in chunks[1:]:
            trimmed += c

        # convert back to numpy
        trimmed_samples = np.array(trimmed.get_array_of_samples()).astype(np.float32)
        trimmed_samples = trimmed_samples / (2 ** 15)  # normalize back to [-1, 1]
        return trimmed_samples

    except Exception:
        return audio_array  # fallback if trimming fails


# -------------------------------------------------------------------
# TEXT NORMALIZATION FOR DATASET
# -------------------------------------------------------------------

def normalize_text_columns(ds):
    cols = ds.column_names
    if "transcription" in cols:
        ds = ds.rename_column("transcription", "text")
    elif "sentence" in cols:
        ds = ds.rename_column("sentence", "text")
    elif "text" not in cols:
        print(f"⚠️ No usable text column: {cols}")
        return None

    for bad in ["label", "client_id", "path", "variant", "segment"]:
        if bad in ds.column_names:
            ds = ds.remove_columns(bad)

    return ds


# -------------------------------------------------------------------
# LOAD JSON DATASET
# -------------------------------------------------------------------

def load_json_data(json_paths):
    datasets = []
    for json_file in json_paths:
        if not json_file.exists():
            print(f"⚠️ Missing file: {json_file}")
            continue

        print(f"📥 Loading {json_file.name}")
        ds = load_dataset("json", data_files=str(json_file), split="train")

        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        ds = normalize_text_columns(ds)

        if ds is not None:
            datasets.append(ds)

    return concatenate_datasets(datasets) if datasets else None


# -------------------------------------------------------------------
# 🧩 FINAL BATCH PROCESSOR FOR WHISPER
# -------------------------------------------------------------------

def prepare_batch(batch):
    cleaned_audio = []
    for audio in batch["audio"]:
        arr = audio["array"]
        arr = trim_audio(arr, sampling_rate=16000)
        cleaned_audio.append(arr)

    cleaned_text = [clean_text(t) for t in batch["text"]]

    inputs = processor(
        cleaned_audio,
        sampling_rate=16000,
        text=cleaned_text,
        return_tensors="pt",
        padding="longest",
    )

    batch["input_features"] = inputs.input_features
    batch["labels"] = inputs.labels
    return batch


# -------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    for lang, cfg in LANGS.items():
        ds = load_json_data(cfg["json_files"])
        if ds is None:
            continue

        print(f"🔧 Preprocessing → {lang}")

        ds = ds.map(
            prepare_batch,
            batched=True,
            batch_size=8,
            num_proc=1,
            remove_columns=[c for c in ds.column_names if c not in ["audio", "text"]],
            load_from_cache_file=False,
            desc=f"🎙️ Cleaning & Feature Extraction [{lang}]",
        )

        out_dir = SAVE_DIR / lang
        ds.save_to_disk(str(out_dir), num_proc=1)

        print(f"💾 Saved Whisper-ready dataset → {out_dir}")

        del ds
        import gc; gc.collect()
