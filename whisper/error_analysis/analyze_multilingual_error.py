import json
import logging
from pathlib import Path
from jiwer import wer, cer
from argparse import ArgumentParser
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path setup (don't change)
BASE_DIR = Path(__file__).resolve().parent.parent   # goes to wav2vec2/
INFERENCE_DIR = BASE_DIR / "error_analysis"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def normalize_text(text):
    """
    Normalize text
    """
    return re.sub(r'\[.*?\]', '', text.lower().strip())


def compute_wer_cer(references, hypothesis):
    """
    Compute Word Error Rate (WER) and Character Error Rate (CER)
    """
    return {
        "wer": wer(references, hypothesis),
        "cer": cer(references, hypothesis)
    }


def evaluate_file(file_path):
    references = []
    hypotheses = []
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            for line in f:
                data = json.loads(line)
                ref = normalize_text(data['reference'])
                hyp = normalize_text(data['hypothesis'])

                if not ref or not hyp:
                    continue

                references.append(ref)
                hypotheses.append(hyp)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON in file {file_path}: {e}")
            return None 
    metrics = compute_wer_cer(hypotheses, references)

    return metrics



def analyze_language(language_code: str):
    """
    Evaluates all files starting with the language name, saves WER & CER summary to /results.
    """
    logger.info(f"🔍 Analyzing inference results for: {language_code}")
    all_results = {}

    # Match pattern: ndebele*.json
    for file in sorted(INFERENCE_DIR.glob(f"{language_code}*.json")):
        name_key = file.stem.replace("_inference", "")
        result = evaluate_file(file)
        if result:
            all_results[name_key] = result
        else:
            logger.warning(f"⚠ Skipped file due to empty or invalid format: {file.name}")

    # Save to /results
    output_file = RESULTS_DIR / f"{language_code}_analysis_results.json"
    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(all_results, out, indent=2, ensure_ascii=False)

    logger.info(f"✅ Saved WER/CER results to: {output_file.name}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--language", required=True, help="Language prefix (e.g., ndebele, siswati)")
    args = parser.parse_args()
    analyze_language(args.language.strip().lower())
