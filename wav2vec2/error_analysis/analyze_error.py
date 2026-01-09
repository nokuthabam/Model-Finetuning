import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from jiwer import wer, cer
import re
import logging
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR /"logs" / "error_analysis"
OUTPUT_DIR = BASE_DIR / "results" / "error_analysis"
ERROR_ANALYSIS_DIR = BASE_DIR / "error_analysis"

# Make sure the output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGE_MAP = {
    "zu": "zulu",
    "xh": "xhosa",
    "ssw": "siswati",
    "nbl": "ndebele"
}

def setup_logging(language_code):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{language_code}_error_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger



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


def character_substitution_analysis(references, hypothesis):
    """
    Analyze character substitutions in the hypothesis compared to the reference.
    """
    reference_characters = list(references.replace(" ", ""))
    hypothesis_characters = list(hypothesis.replace(" ", ""))

    substitutions = Counter()
    for ref, hyp in zip(reference_characters, hypothesis_characters):
        if ref != hyp:
            substitutions[(ref, hyp)] += 1
    return substitutions


def analyze_morphological_errors(references, hypothesis):
    """
    Analyze morphological errors in the hypothesis compared to the reference.
    """
    reference_tokens = references.split()
    hypothesis_tokens = hypothesis.split()
    errors = []
    misaligned = len(reference_tokens) != len(hypothesis_tokens)

    for ref_token, hyp_token in zip(reference_tokens, hypothesis_tokens):
        if ref_token != hyp_token:
            errors.append((ref_token, hyp_token))

    return {
        "misaligned": misaligned,
        "errors": errors
    }


def analyze_file(language_code, logger):
    """
    Analyze a single file for error analysis.
    """

    language = LANGUAGE_MAP.get(language_code, language_code)
    path = ERROR_ANALYSIS_DIR / f"{language}_inference_results.json"
    logger.info(f"Loading inference results for {language} from {path}")

    all_substitutions = Counter()
    hypothesis_texts = []
    reference_texts = []
    problematic_texts = []

    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            ref = normalize_text(data['reference'])
            hyp = normalize_text(data['hypothesis'])

            if not ref or not hyp:
                continue

            reference_texts.append(ref)
            hypothesis_texts.append(hyp)

            if ref != hyp:
                subs = character_substitution_analysis(ref, hyp)
                morphs = analyze_morphological_errors(ref, hyp)
                all_substitutions.update(subs)

                if len(ref.split()) < 6:
                    problematic_texts.append(
                        {
                            "reference": ref,
                            "hypothesis": hyp,
                            "subs": {f"{a} → {b}": count for (a, b), count in subs.items()},
                            "morphological_errors": morphs
                        }
                )

    metrics = compute_wer_cer(hypothesis_texts, reference_texts)
    report_path = OUTPUT_DIR / f"{language}_error_analysis_report.json"

    substitution_list = [
        {"substitution": f"{a} → {b}", "count": count}
        for (a, b), count in all_substitutions.most_common(10)
    ]
    print(substitution_list)

    # Build report dict
    report = {
        "language": language,
        "metrics": metrics,
        "phoneme_substitutions": substitution_list,
        "examples": problematic_texts,
        "notes": {
            "Substitution note": "These are character level substitutions which are used to approximate phoneme confusion.",
            "Morphological analysis note": "Morphological errors include stem mismatches and prefix/suffix mismatches."
        }
    }

    # Save as JSON
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Error analysis report saved to {report_path}")
    logger.info(f"Top 10 phoneme substitutions: {all_substitutions.most_common(10)}")
    logger.info(f"WER: {metrics['wer']}, CER: {metrics['cer']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, help="Language code for error analysis", required=True, choices=LANGUAGE_MAP.keys())
    args = parser.parse_args()

    logger = setup_logging(args.language)
    analyze_file(args.language, logger)