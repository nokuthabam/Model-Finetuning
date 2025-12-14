# e:/Model-Finetuning/whisper/model/preprocess_multilingual.py

from datasets import load_dataset, Audio, concatenate_datasets
from transformers import WhisperProcessor
from pathlib import Path
from pydub import AudioSegment, silence
import numpy as np
import re, unicodedata
import argparse
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

# -----------------------------------------
# CONFIG
# -----------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAVE_DIR = BASE_DIR / "model" / "processed_arrow" / "multilingual"
MODEL_NAME = "openai/whisper-small"

LANG_JSON_MAP = {
    "zu": BASE_DIR / "merged_zu.json",
    "xh": BASE_DIR / "merged_xh.json",
    "ss": BASE_DIR / "merged_ss.json",
    "nr": BASE_DIR / "merged_nr.json",
}

processor = WhisperProcessor.from_pretrained(MODEL_NAME)

# -----------------------------------------
# TEXT CLEANING
# -----------------------------------------
def clean_text(text: str):
    if not isinstance(text, str):
        return ""

    text = re.sub(r"\[([^]]+)\]", r"\1", text)  # remove [..] but keep content
    text = text.replace("[", "").replace("]", "")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

# -----------------------------------------
# SILENCE TRIMMING
# -----------------------------------------
def trim_audio(audio_array, sampling_rate=16000, top_db=35):
    try:
        sound = AudioSegment(
            audio_array.tobytes(),
            frame_rate=sampling_rate,
            sample_width=audio_array.dtype.itemsize,
            channels=1
        )

        chunks = silence.split_on_silence(
            sound,
            min_silence_len=300,
            silence_thresh=sound.dBFS - top_db,
            keep_silence=150,
        )

        if len(chunks) == 0:
            return audio_array

        trimmed = chunks[0]
        for c in chunks[1:]:
            trimmed += c

        trimmed_samples = np.array(trimmed.get_array_of_samples()).astype(np.float32)
        trimmed_samples = trimmed_samples / (2 ** 15)
        return trimmed_samples

    except Exception:
        return audio_array

# -----------------------------------------
# LOAD JSON
# -----------------------------------------
def load_json_dataset(json_file: Path):
    print(f"📥 Loading {json_file.name}")
    ds = load_dataset("json", data_files=str(json_file), split="train")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # Normalizing text column
    if "transcription" in ds.column_names:
        ds = ds.rename_column("transcription", "text")
    elif "sentence" in ds.column_names:
        ds = ds.rename_column("sentence", "text")

    keep = ["audio", "text"]
    drop_cols = [c for c in ds.column_names if c not in keep]
    ds = ds.remove_columns(drop_cols)

    return ds

# -----------------------------------------
# FINAL BATCH PROCESSOR
# -----------------------------------------
def prepare_batch(batch):
    cleaned_audio = [trim_audio(a["array"]) for a in batch["audio"]]
    cleaned_text = [clean_text(t) for t in batch["text"]]

    inputs = processor(
        cleaned_audio,
        sampling_rate=16000,
        text=cleaned_text,
        return_tensors="pt",
        padding="longest",
    )

    return {
        "input_features": inputs.input_features,
        "labels": inputs.labels,
        "audio": batch["audio"],
        "text": cleaned_text,
    }

# -----------------------------------------
# MAIN
# -----------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--languages",
        nargs="+",
        required=True,
        help="Languages to include: zu xh ss nr",
    )
    args = parser.parse_args()

    langs = args.languages
    for L in langs:
        if L not in LANG_JSON_MAP:
            raise ValueError(f"❌ Unknown language code: {L}")

    multilingual_name = "_".join(langs)   #nguni_multilingual_langs_whisper
    multilingual_name = "nguni_multilingual_" + multilingual_name + "_whisper"
    output_dir = SAVE_DIR / multilingual_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Load + combine datasets
    # ----------------------------
    datasets = []
    for L in langs:
        ds = load_json_dataset(LANG_JSON_MAP[L])
        datasets.append(ds)

    merged = concatenate_datasets(datasets)

    print(f"🔧 Preprocessing multilingual dataset: {langs}")
    merged = merged.map(
        prepare_batch,
        batched=True,
        batch_size=8,
        num_proc=1,
        load_from_cache_file=False,
        desc=f"🎙️ Cleaning + Feature Extraction [{multilingual_name}]",
    )

    merged.save_to_disk(str(output_dir), num_proc=1)

    print(f"💾 Saved multilingual dataset → {output_dir}")
