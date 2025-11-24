import json
import os
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
JSON_DIR = BASE_DIR / "error_analysis" / "training_metrics"
PLOTS_DIR = BASE_DIR / "results" / "training_metrics_plots"


def load_wer_from_trainer_state(trainer_state_path):
    with open(trainer_state_path, "r") as f:
        data = json.load(f)
    log_history = data.get("log_history", [])

    steps = []
    wers = []

    for entry in log_history:
        if "eval_wer" in entry:
            steps.append(entry.get("step", None))
            wers.append(entry.get("eval_wer"))

    return steps, wers


def plot_wer(lang_dir, lang_code):
    lang_dir = Path(lang_dir)
    trainer_state_path = lang_dir / "trainer_state.json"

    if not trainer_state_path.exists():
        raise FileNotFoundError(f"No trainer_state.json found in: {trainer_state_path}")

    steps, wers = load_wer_from_trainer_state(trainer_state_path)

    if not steps:
        raise ValueError("Could not find eval_wer values in trainer_state.json")

    plt.figure(figsize=(8, 5))
    plt.plot(steps, wers, marker='o', label=f"{lang_code.upper()} WER")

    plt.title(f"WER vs Steps — {lang_code.upper()} Whisper Small")
    plt.xlabel("Training Steps")
    plt.ylabel("WER (%)")
    plt.grid(True)
    plt.legend()

    out_path = lang_dir / f"{lang_code}_wer_curve.png"
    plt.savefig(out_path, dpi=300)
    print(f"📊 Saved plot to {out_path}")


if __name__ == "__main__":
    # Example usage:
    # python plot_wer_curve.py --lang_dir model/ss_whisper --lang ss
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_dir", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)

    args = parser.parse_args()
    plot_wer(args.lang_dir, args.lang)