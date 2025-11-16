# e:/Model-Finetuning/whisper/model/preprocess.py
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import WhisperProcessor
from pathlib import Path
from pydub import AudioSegment, silence
import numpy as np
import os

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAVE_DIR = BASE_DIR / "model" / "processed_arrow"
MODEL_NAME = "openai/whisper-small"

LANGS = {
    # "zulu": {
    #     "json_files": [
    #         DATA_DIR / "zulu_dataset.json",
    #         DATA_DIR / "nchlt_zu_whisper.json",
    #         DATA_DIR / "commonvoice_zu_train.json",
    #     ]
    # },
    # "xhosa": {
    #     "json_files": [
    #         DATA_DIR / "xhosa_dataset.json",
    #         DATA_DIR / "nchlt_xh_whisper.json",
    #         DATA_DIR / "commonvoice_xh_train.json",
    #     ]
    # },
    # "ndebele": {
    #     "json_files": [
    #         DATA_DIR / "ndebele_dataset.json",
    #         DATA_DIR / "nchlt_nr_whisper.json",
    #     ]
    # },
    # "siswati": {
    #     "json_files": [
    #         DATA_DIR / "siswati_dataset.json",
    #         DATA_DIR / "nchlt_ss_whisper.json",
    #     ]
    # },
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
    elif "text" not in cols:
        print(f"⚠️ No text column found. Columns: {cols}")
        return None

    for bad_col in ["label", "client_id", "path", "variant", "segment"]:
        if bad_col in ds.column_names:
            ds = ds.remove_columns(bad_col)

    return ds

# -------------------------------------------------------------------
# LOAD + CLEAN
# -------------------------------------------------------------------
def load_json_data(json_paths):
    datasets = []
    for json_file in json_paths:
        if not json_file.exists():
            print(f"⚠️ Skipping missing file: {json_file}")
            continue

        print(f"✅ Loading {json_file.name}")
        ds = load_dataset("json", data_files=str(json_file), split="train")
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        ds = normalize_text_columns(ds)
        if ds is not None:
            datasets.append(ds)

    return concatenate_datasets(datasets) if datasets else None

# -------------------------------------------------------------------
# PREPARE FOR WHISPER
# -------------------------------------------------------------------
def prepare_batch(batch):
    audio = batch["audio"]
    text = batch["text"]

    if audio is None or audio.get("array") is None:
        raise ValueError("Missing audio array")

    inputs = processor(
        audio["array"],
        sampling_rate=16000,
        text=text,
        return_tensors="pt",
        padding="longest",
    )
    batch["input_features"] = inputs.input_features[0]
    batch["labels"] = inputs.labels[0]
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

        print(f"🔧 Preprocessing + silence trimming → {lang}")
        keep = ["audio", "text"]
        cols_to_remove = [c for c in ds.column_names if c not in keep]

        ds = ds.map(
            prepare_batch,
            remove_columns=cols_to_remove,
            num_proc=1,
            desc=f"🎙️ Processing {lang}",
        )

        lang_dir = SAVE_DIR / lang
        ds.save_to_disk(str(lang_dir))
        print(f"💾 Saved Whisper-ready {lang} dataset → {lang_dir}")
