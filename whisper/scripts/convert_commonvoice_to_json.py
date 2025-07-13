import pandas as pd
import json
from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent.parent

def clean_text(text):
    """
    Lowercases and removes unwanted punctuation, keeping brackets and apostrophes.
    """
    return re.sub(r"[^\w\s'\[\]]", "", text.lower().strip())

def convert_commonvoice_tsv_to_jsonl(tsv_path, clips_dir, output_json_path):
    """
    Converts a Common Voice .tsv split to Lwazi-style JSONL format.
    
    Args:
        tsv_path (str or Path): Path to Common Voice .tsv file
        clips_dir (str or Path): Path to 'clips/' directory containing .mp3 files
        output_json_path (str or Path): Path to save the resulting JSONL file
    """
    tsv_path = Path(tsv_path)
    clips_dir = Path(clips_dir)
    output_json_path = Path(output_json_path)

    df = pd.read_csv(tsv_path, sep="\t")
    
    with open(output_json_path, "w", encoding="utf-8") as out_file:
        for _, row in df.iterrows():
            audio_path = clips_dir / row["path"]
            entry = {
                "audio_path": str(audio_path),
                "transcript": clean_text(row["sentence"]),
                "speaker_id": row.get("client_id", "unknown"),
                "age": row.get("age", "unknown")
            }
            out_file.write(json.dumps(entry) + "\n")
    print(f"✅ Saved {len(df)} entries to {output_json_path}")

# Example usage:
if __name__ == "__main__":
    convert_commonvoice_tsv_to_jsonl(
        tsv_path=BASE_DIR/"data/commonvoice_zu_merged/train.tsv",
        clips_dir=BASE_DIR/"data/commonvoice_zu_merged/clips",
        output_json_path=BASE_DIR/"data/commonvoice_zu_train.json"
    )

    convert_commonvoice_tsv_to_jsonl(
        tsv_path=BASE_DIR/"data/commonvoice_xh_merged/train.tsv",
        clips_dir=BASE_DIR/"data/commonvoice_xh_merged/clips",
        output_json_path=BASE_DIR/"data/commonvoice_xh_train.json"
    )
