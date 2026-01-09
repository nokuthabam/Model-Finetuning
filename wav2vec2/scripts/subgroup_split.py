from importlib import metadata
import os
import json
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime


LANGUAGES = {
    "zulu": "zul",
    "xhosa": "xho",
    "siswati": "ssw",
    "ndebele": "nbl"
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
    Classify age into groups.
    """
    if age is None:
        return "unknown"

    age = str(age).strip().lower()

    # Handle common non-numeric cases
    if age in {"nan", "not available", "unknown", "", "-1"}:
        return "unknown"

    # Handle decades like "70s"
    if age.endswith("s") and age[:-1].isdigit():
        age = int(age[:-1])
    else:
        try:
            age = int(age)
        except ValueError:
            return "unknown"
    if age <= 0:
        return "unknown"
    elif age <= 18:
        return "18_below"
    elif age < 30:
        return "19_29"
    elif age < 40:
        return "30_39"
    elif age < 50:
        return "40_49"
    else:
        return "50_plus"


def load_inference_results(json_path):
    """
    Load inference results from a JSON file.
    """
    with open(json_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data


def split_by_age_and_gender(entries):
    age_groups = {}
    gender_groups = {}
    for entry in entries:
        age = entry.get("age")
        gender = str(entry.get("gender", "unknown")).lower()
        age_group = age_group_def(age)

        age_groups.setdefault(age_group, []).append(entry)
        gender_groups.setdefault(gender, []).append(entry)
    
    return age_groups, gender_groups


def write_outputs(base_name, output_dir, grouped_data):
    for group, entries in grouped_data.items():
        output_path = output_dir / f"{base_name}_{group}.json"
        with open(output_path, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        logging.info(f"Wrote {len(entries)} entries to {output_path}")


def load_test_data(language, language_code):
    # Lwazi
    files = [
        DATA_DIR / f"{language}_dataset.json",
        DATA_DIR / f"nchlt_{language_code}.json"
    ]

    data = []
    for filepath in files:
        if not filepath.exists():
            logging.warning(f"Test data file {filepath} does not exist.")
            continue
        logging.info(f"Loading test data from {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                entry["dataset"] = "nchlt" if "nchlt" in filepath.name else "lwazi"
                data.append(entry)
    return data


def add_metadata(inference_entries, test_entries):
    meta_data = {}
    for entry in test_entries:
        meta_data[entry['audio_path']] = {
            "age": entry.get("age"),
            "gender": entry.get("gender") or entry.get("sex"),
        }

    final = []
    for entry in inference_entries:
        meta = meta_data.get(entry['audio_path'], {})
        entry["age"] = meta.get("age")
        entry["gender"] = meta.get("gender", "unknown")
        final.append(entry)
    return final


def main():
    logger = logging.getLogger(__name__)
    setup_logging()
    for lang, lang_code in LANGUAGES.items():
        logger.info("=" * 50)
        logger.info(f"Processing language: {lang}")
        logger.info("=" * 50)

        # Load inference results
        inference_path = OUTPUT_DIR / f"{lang}_inference_results.json"
        if not inference_path.exists():
            logger.error(f"Inference results file {inference_path} does not exist. Skipping.")
            continue
        
        inference_entries = load_inference_results(inference_path)
        logger.info(f"Loaded {len(inference_entries)} inference entries for language: {lang}")
        print(f"Loaded {len(inference_entries)} inference entries for language: {lang}")

        test_entries = load_test_data(lang, lang_code)
        logger.info(f"Loaded {len(test_entries)} test entries for language: {lang}")
        print(f"Loaded {len(test_entries)} test entries for language: {lang}")

        combined_entries = add_metadata(inference_entries, test_entries)
        logger.info(f"Combined entries count: {len(combined_entries)}")
        print(f"Combined entries count: {len(combined_entries)}")

        age_groups, gender_groups = split_by_age_and_gender(combined_entries)

        write_outputs(lang, OUTPUT_DIR, age_groups)
        write_outputs(lang, OUTPUT_DIR, gender_groups)

        logger.info(f"Completed processing for language: {lang}")
        print(f"Completed processing for language: {lang}")
    logger.info("All processing complete.")


if __name__ == "__main__":
    main()