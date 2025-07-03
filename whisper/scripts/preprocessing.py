import os
import pandas as pd
import json
from pathlib import Path
import re


#Paths
BASE_DIR = Path(__file__).resolve().parent.parent
ZULU_TRANSCRIPT_PATH = BASE_DIR/ "data/ASR.Lwazi.Zul.1.0/transcriptions.csv"
ZULU_METADATA_PATH = BASE_DIR/ "data/ASR.Lwazi.Zul.1.0/Lwazi_metadata_isizulu.csv"
ZULU_AUDIO_BASE_DIR = BASE_DIR / "data/ASR.Lwazi.Zul.1.0/audio"
XHOSA_TRANSCRIPT_PATH = BASE_DIR / "data/ASR.Lwazi.Xho.1.0/transcriptions.csv"
XHOSA_METADATA_PATH = BASE_DIR / "data/ASR.Lwazi.Xho.1.0/Lwazi_metadata_isixhosa.csv"
XHOSA_AUDIO_BASE_DIR = BASE_DIR / "data/ASR.Lwazi.Xho.1.0/audio"
NDEBELE_TRANSCRIPT_PATH = BASE_DIR / "data/ASR.Lwazi.Nbl.1.0/transcriptions.csv"
NDEBELE_METADATA_PATH = BASE_DIR / "data/ASR.Lwazi.Nbl.1.0/Lwazi_metadata_isindebele.csv"
NDEBELE_AUDIO_BASE_DIR = BASE_DIR / "data/ASR.Lwazi.Nbl.1.0/audio"  
SISWATI_TRANSCRIPT_PATH = BASE_DIR / "data/ASR.Lwazi.Ssw.1.0/transcriptions.csv"
SISWATI_METADATA_PATH = BASE_DIR / "data/ASR.Lwazi.Ssw.1.0/Lwazi_metadata_siswati.csv"
SISWATI_AUDIO_BASE_DIR = BASE_DIR / "data/ASR.Lwazi.Ssw.1.0/audio"

def load_speaker_metadata(metadata_path, language):
    """
    Load speaker metadata from a CSV file.
    """
    metadata = pd.read_csv(metadata_path, skiprows=1)
    metadata.columns = ["Speaker", "Gender", "LineType", "Age"]
    return {
        f"{language}_{int(row['Speaker']):03d}": row["Age"]
        for _, row in metadata.iterrows()
    }


def clean_transcript(transcript):
    """
    Clean the transcript text by converting to lowercase,
    removing unwanted characters and leading or trailing spaces
    """
    transcript = transcript.lower().strip()
    transcript = re.sub(r"[^\w\s'\[\]]", "", transcript)
    return transcript


def build_unified_dataset(transcript_path, speaker_age_map, audio_base_dir):
    """
    Build a unified dataset from the transcript and metadata files.
    """
    df = pd.read_csv(transcript_path)
    unified_dataset = []
    for _, row in df.iterrows():
        subfolder = row["Subfolder"]
        file_id = row["File ID"]
        audio_path = audio_base_dir / subfolder / f"{file_id}.wav"
        age = speaker_age_map.get(subfolder, "Unkown")
        transcript = clean_transcript(row["Transcription"])
        unified_dataset.append({
            "audio_path": str(audio_path),
            "transcript": transcript,
            "speaker_id": subfolder,
            "age": age
        })
    return unified_dataset
    

def main():
    
    # Zulu Logic
    speaker_age_map_zulu = load_speaker_metadata(ZULU_METADATA_PATH, "isizulu")
    zulu_dataset = build_unified_dataset(ZULU_TRANSCRIPT_PATH, speaker_age_map_zulu, ZULU_AUDIO_BASE_DIR)
    with open(BASE_DIR / "data/zulu_dataset.json", "w") as f:
        for item in zulu_dataset:
            f.write(json.dumps(item) + "\n")
    print(f"Zulu dataset created with {len(zulu_dataset)} entries.")

    # Xhosa Logic
    speaker_age_map_xhosa = load_speaker_metadata(XHOSA_METADATA_PATH, "isixhosa")
    xhosa_dataset = build_unified_dataset(XHOSA_TRANSCRIPT_PATH, speaker_age_map_xhosa, XHOSA_AUDIO_BASE_DIR)
    with open(BASE_DIR / "data/xhosa_dataset.json", "w") as f:
        for item in xhosa_dataset:
            f.write(json.dumps(item) + "\n")
    print(f"Xhosa dataset created with {len(xhosa_dataset)} entries.")

    # Ndebele Logic
    speaker_age_map_ndebele = load_speaker_metadata(NDEBELE_METADATA_PATH , "isindebele")
    ndebele_dataset = build_unified_dataset(NDEBELE_TRANSCRIPT_PATH, speaker_age_map_ndebele, NDEBELE_AUDIO_BASE_DIR)
    with open(BASE_DIR / "data/ndebele_dataset.json", "w") as f:
        for item in ndebele_dataset:
            f.write(json.dumps(item) + "\n")
    print(f"Ndebele dataset created with {len(ndebele_dataset)} entries.")

    # Siswati Logic
    speaker_age_map_siswati = load_speaker_metadata(SISWATI_METADATA_PATH, "siswati")
    siswati_dataset = build_unified_dataset(SISWATI_TRANSCRIPT_PATH, speaker_age_map_siswati, SISWATI_AUDIO_BASE_DIR)
    with open(BASE_DIR / "data/siswati_dataset.json", "w") as f:
        for item in siswati_dataset:
            f.write(json.dumps(item) + "\n")
    print(f"Siswati dataset created with {len(siswati_dataset)} entries.")


if __name__ == "__main__":
    main()
    
    