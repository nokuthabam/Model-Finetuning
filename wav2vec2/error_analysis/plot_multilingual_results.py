import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
ANALYSIS_DIR = BASE_DIR / "error_analysis"


def plot_multilingual_results(language: str):
    """
    Plot WER and CER for multiple languages
    """

    input_file = RESULTS_DIR / f"{language}_analysis_results.json"
    output_file = RESULTS_DIR / f"{language}_wer_cer_plot.png"

    with open(input_file, "r") as f:
        data = json.load(f)

    models = list(data.keys())
    wers = [data[model]["wer"] for model in models]
    cers = [data[model]["cer"] for model in models]

    x = range(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width/2 for i in x], wers, width, label='WER', color='skyblue')
    ax.bar([i + width/2 for i in x], cers, width, label='CER', color='salmon')

    ax.set_ylabel('Error Rate')
    ax.set_title(f'WER and CER for {language} by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot WER and CER for multiple languages")
    parser.add_argument("--language", type=str, required=True, help="Language (e.g., zulu, xhosa, siswati, ndebele)")
    args = parser.parse_args()

    plot_multilingual_results(args.language)
