from datasets import load_dataset, Audio, concatenate_datasets, Value, DatasetDict
from pathlib import Path
import os
import json

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAVE_DIR = BASE_DIR / "processed_arrow"
os.makedirs(SAVE_DIR, exist_ok=True)

LANGS = {
    "zulu": {
        "json_files": [
            DATA_DIR / "zulu_dataset.json",
            DATA_DIR / "nchlt_zul_fixedpaths.json",
            DATA_DIR / "commonvoice_zu_train.json",
        ]
    },
    "xhosa": {
        "json_files": [
            DATA_DIR / "xhosa_dataset.json",
            DATA_DIR / "nchlt_xho_fixedpaths.json",
            DATA_DIR / "commonvoice_xh_train.json",
        ]
    },
    "ndebele": {
        "json_files": [
            DATA_DIR / "ndebele_dataset.json",
            DATA_DIR / "nchlt_nbl_fixedpaths.json",
        ]
    },
    "siswati": {
        "json_files": [
            DATA_DIR / "siswati_dataset.json",
            DATA_DIR / "nchlt_ssw_fixedpaths.json",
        ]
    },
}

# ------------------------------------------------------------
# LOAD JSON FUNCTION
# ------------------------------------------------------------
def load_json_dataset(json_path):
    """Loads a JSON lines file into a Hugging Face Dataset."""
    if not Path(json_path).exists():
        print(f"⚠️ Missing {json_path}, skipping...")
        return None
    return load_dataset("json", data_files=json_path, split="train")


# ------------------------------------------------------------
# NORMALIZE TEXT COLUMN
# ------------------------------------------------------------
def normalize_text(batch):
    """Unifies text field names and ensures lowercase normalized text."""
    text_fields = ["text", "transcript", "sentence", "transcription"]
    for key in text_fields:
        if key in batch and batch[key]:
            batch["text"] = str(batch[key]).strip().lower()
            break
    if "text" not in batch:
        batch["text"] = ""
    return batch


# ------------------------------------------------------------
# MAIN PREPROCESSING LOOP
# ------------------------------------------------------------
def preprocess_language(lang, lang_data):
    print(f"\n🌍 Processing {lang.upper()}...")

    datasets = []
    for path in lang_data["json_files"]:
        if not path.exists():
            print(f"⚠️ Skipping {path.name} (not found)")
            continue

        print(f"📂 Loading {path.name} ...")
        ds = load_json_dataset(str(path))
        ds = ds.map(normalize_text)

        # Handle audio column
        if "audio" not in ds.column_names and "audio_path" in ds.column_names:
            ds = ds.cast_column("audio_path", Audio(sampling_rate=16000))
            ds = ds.rename_column("audio_path", "audio")
        elif "audio" in ds.column_names:
            ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        else:
            print(f"❌ No audio column found for {path.name}, skipping.")
            continue

        # Ensure we have a text column
        if "text" not in ds.column_names:
            print(f"⚠️ No text column found for {path.name}, skipping this file.")
            continue

        # 🧩 Fix 'age' before adding it to list
        if "age" in ds.column_names:
            try:
                ds = ds.cast_column("age", Value("string"))
            except Exception as e:
                print(f"⚠️ Could not cast 'age' column in {path.name}: {e}")
                # fallback: drop the column if casting fails
                ds = ds.remove_columns(["age"])

        datasets.append(ds)

    if not datasets:
        print(f"⚠️ No valid datasets for {lang}, skipping.")
        return

    # 🧠 Now all 'age' columns are harmonized, safe to merge
    combined = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]

    # Final cleanup: rename transcript → text if needed
    if "transcript" in combined.column_names and "text" not in combined.column_names:
        combined = combined.rename_column("transcript", "text")

    # Save to disk in Arrow format
    save_path = SAVE_DIR / lang
    combined.save_to_disk(str(save_path))
    print(f"💾 Saved {lang} dataset → {save_path}")


# ------------------------------------------------------------
# RUN ALL LANGUAGES
# ------------------------------------------------------------
if __name__ == "__main__":
    for lang, lang_data in LANGS.items():
        preprocess_language(lang, lang_data)