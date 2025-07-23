import json
from pathlib import Path
import logging
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MANIFESTS_DIR = BASE_DIR / "manifests"
LOG_DIR = BASE_DIR / "logs"

LANGUAGES = {
    "zu": "zulu",
    "xh": "xhosa",
    "ss": "siswati",
    "nr": "ndebele",
}

def setup_logging(language_code):
    """
    Set up logging configuration.
    """
    log_dir = LOG_DIR / language_code
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"generate_transcripts_{language_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    return logging.getLogger(__name__)


def extract_transcripts(json_path):
    """
    Extract transcripts from the manifest files for the specified language.
    """
    transcripts = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            if 'transcript' in entry:
                transcripts.append(entry['transcript'])
    return transcripts


def write_text_file(language_code, transcripts):
    """
    Write the transcripts to a text file.
    """
    lang_name = LANGUAGES.get(language_code, language_code)
    text_file_path = MANIFESTS_DIR / f"{lang_name}_text.txt"
    with open(text_file_path, 'w', encoding='utf-8') as f:
        for transcript in transcripts:
            f.write(transcript + '\n')
    return text_file_path


def main():
    logger = setup_logging("transcripts")
    logger.info("Starting transcript generation script.")
    for lang_code, lang_name in LANGUAGES.items():
        transcripts = []

        # Lwazi file
        lwazi_json_path = DATA_DIR / f"{lang_name}_train.json"
        if lwazi_json_path.exists():
            logger.info(f"Processing Lwazi file for {lang_name}.")
            transcripts += extract_transcripts(lwazi_json_path)
        # Common Voice file
        cv_file = DATA_DIR / f"commonvoice_{lang_code}_train.json"
        if cv_file.exists():
            logger.info(f"Processing Common Voice file for {lang_name}.")
            transcripts += extract_transcripts(cv_file)
        if transcripts:
            write_text_file(lang_code, transcripts)
            logger.info(f"Transcripts for {lang_name} written to text file.")
        else:
            logger.warning(f"No transcripts found for {lang_name}.")


if __name__ == "__main__":
    main()