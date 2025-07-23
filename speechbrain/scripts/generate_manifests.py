import json
import csv
import torchaudio
from pathlib import Path
import logging
from datetime import datetime

LANGUAGES = {
    "zu": "Zulu",
    "xh": "Xhosa",
    "ss": "Siswati",
    "nr": "Ndebele",
}

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MANIFESTS_DIR = BASE_DIR / "manifests"
LOG_DIR = BASE_DIR / "logs"

def setup_logging(language_code):
    """
    Set up logging configuration.
    """
    log_dir = LOG_DIR / language_code
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{language_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def compute_duration(audio_path):
    """
    Compute the duration of an audio file.
    """
    try:
        info = torchaudio.info(audio_path)
        return round(info.num_frames / info.sample_rate, 2)
    except Exception as e:
        print(f"Error reading {audio_path}: {e}")
        return 0.0
    

def json_to_csv(json_file, csv_file):
    """
    Convert a JSON file to a CSV file.
    """
    with open(json_file, "r", encoding="utf-8") as json_file:
        data = [json.loads(line) for line in json_file if line.strip()]

    with open(csv_file, "w", newline='', encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["ID", "wav", "duration", "spk_id", "age", "transcript"])
        writer.writeheader()

        for i, sample in enumerate(data):
            audio_path = sample["audio_path"]
            duration = compute_duration(audio_path)
            writer.writerow({
                "ID": f"{Path(audio_path).stem}_{i}",
                "wav": audio_path,
                "duration": duration,
                "spk_id": sample.get("speaker_id", "unknown"),
                "age": sample.get("age", "unknown"),
                "transcript": sample["transcript"]
            })

def main():

    logger = setup_logging("default")
    logger.info("Starting manifest generation...")

    for code, language, in LANGUAGES.items():
        for split in ["train", "test"]:
            json_file = DATA_DIR / f"{language.lower()}_{split}.json"
            csv_file = MANIFESTS_DIR / f"{language.lower()}_{split}.csv"
            if json_file.exists():
                logger.info(f"Converting {json_file} to {csv_file}")
                json_to_csv(json_file, csv_file)
            else:
                logger.warning(f"File {json_file} does not exist.")

        logger.info(f"Completed manifest generation for {language}.")

        language_files = {
        "commonvoice_zu_train": "commonvoice_zu_train.json",
        "commonvoice_xh_train": "commonvoice_xh_train.json"
    }

    for name, file in language_files.items():
        json_file = DATA_DIR / file
        csv_file = MANIFESTS_DIR / f"{name}.csv"
        if json_file.exists():
            logger.info(f"Converting {json_file} to {csv_file}")
            json_to_csv(json_file, csv_file)
        else:
            logger.warning(f"File {json_file} does not exist.")

if __name__ == "__main__":
    main()
