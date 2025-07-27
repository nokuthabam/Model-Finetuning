from importlib import metadata
import os
import json
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime


LANGUAGES = {
    "zulu": "isizulu",
    "xhosa": "isixhosa",
    "siswati": "siswati",
    "ndebele": "isindebele"
}

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "error_analysis"
LOGS = BASE_DIR / "logs"

def setup_logging():
    """
    Set up logging configuration
    """
    if not LOGS.exists():
        LOGS.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        filename= LOGS / f"subgroup_split_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("Logging setup complete.")


def age_group_def(age):
    """
    Classify age into groups
    """
    try:
        age = int(age)
        if age <= 18:
            return "18 below"
        elif age < 30:
            return "19-29"
        elif age < 40:
            return "30-39"
        elif age < 50:
            return "40-49"
        else:
            return "50+"
    except ValueError:
        return "Unknown"


def main():
    logger = logging.getLogger(__name__)
    setup_logging()
    # iterate through each language
    for lang, lang_code in LANGUAGES.items():
        metadata_file = DATA_DIR / f"Lwazi_metadata_{lang_code}.csv"
        if not metadata_file.exists():
            logger.warning(f"Metadata file for {lang} not found.")
            continue

        metadata = pd.read_csv(metadata_file)
        metadata.columns = metadata.iloc[0]  # Set the first row as header
        metadata = metadata[1:] # Remove the first row
        metadata = metadata.rename(columns={"Speaker": "speaker_id", "Gender": "gender", "Age": "age"})
        metadata["speaker_id"] = metadata["speaker_id"].astype(int).apply(lambda x: f"{lang_code}_{x}".lower())
        metadata["age"] = pd.to_numeric(metadata["age"], errors='coerce')
        metadata["age_group"] = metadata["age"].apply(age_group_def)

        speaker_info = metadata.set_index("speaker_id")[["gender", "age_group"]].to_dict(orient="index")

        # load the inference results
        json_path = OUTPUT_DIR / f"{lang}_inference_results.json"
        logger.info(f"Loading inference results from {json_path}")
        if not json_path.exists():
            logger.warning(f"Inference results for {lang} not found.")
            continue
        
        with open(json_path, 'r') as f:
            data = [json.loads(line) for line in f]

        # Organise data into gender and age groups
        gender_groups = {}
        age_groups = {}

        for entry in data:
            speaker_id = Path(entry["audio"]).parts[-2].lower()
            speaker_meta = speaker_info.get(speaker_id)

            if not speaker_meta:
                logger.warning(f"Speaker metadata not found for {speaker_id}. Skipping entry.")
                continue

            gender = str(speaker_meta.get("gender", "unknown")).lower().replace(" ", "_")
            age_group = speaker_meta["age_group"]

            gender_groups.setdefault(gender, []).append(entry)
            age_groups.setdefault(age_group, []).append(entry)

        # Write JSON files by age group
        for gender, entries in gender_groups.items():
            output_path = OUTPUT_DIR / f"{lang}_{gender}.json"
            with open(output_path, 'w') as f:
                for e in entries:
                    f.write(json.dumps(e) + "\n")


        for age_group, entries in age_groups.items():
            age_slug = age_group.replace("+", "plus").replace("-", "_").lower()
            output_path = OUTPUT_DIR / f"{lang}_{age_slug}.json"
            with open(output_path, 'w') as f:
                for e in entries:
                    f.write(json.dumps(e) + "\n")

        print(f"Processed {lang} data: {len(data)} entries")


if __name__ == "__main__":
    main()

