import json
from pathlib import Path
import matplotlib.pyplot as plt
from jiwer import wer, cer

BASE_DIR = Path(__file__).resolve().parent.parent
JSON_DIR = BASE_DIR / "error_analysis"
OUTPUT_DIR = BASE_DIR / "results" / "error_analysis"
LANGUAGES = ["zulu", "xhosa", "siswati", "ndebele"]

age_groups = ["18_below", "19_29", "30_39", "40_49", "50_plus"]

wer_results = {age: [] for age in age_groups}
cer_results = {age: [] for age in age_groups}

for age in age_groups:
    for lang in LANGUAGES:
        json_path = JSON_DIR / f"{lang}_{age}.json"
        if json_path.exists():
            with open(json_path, "r") as f:
                references, hypotheses = [], []
                for line in f:
                    item = json.loads(line)
                    ref = item['reference']
                    hyp = item['hypothesis']
                    references.append(ref)
                    hypotheses.append(hyp)
                wer_results[age].append(wer(references, hypotheses))
                cer_results[age].append(cer(references, hypotheses))
print(f"wer_results: {wer_results}")
print(f"cer_results: {cer_results}")
# use a consistent color palette for the languages
colors = ["#1f76b48b", "#ffe70eb7", "#2ca02c94", "#d6272794"]  # blue, orange, green, red

# Plotting WER results
x = range(len(age_groups))
bar_width = 0.2
plt.figure(figsize=(12, 8))
for i, lang in enumerate(LANGUAGES):
    values = [wer_results[age][i] for age in age_groups]
    positions = [pos + i * bar_width for pos in x]
    plt.bar(positions, values, width=bar_width, label=lang.capitalize(), color=colors[i% len(colors)])

plt.xticks([j + 1.5 * bar_width for j in x], age_groups)
plt.xlabel("Age Groups")
plt.ylabel("Word Error Rate (WER)")
plt.title("Word Error Rate by Age Group and Language - Whisper")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "age_wer.png")
plt.show()


# Plotting CER results
plt.figure(figsize=(12, 8))
for i, lang in enumerate(LANGUAGES):
    values = [cer_results[age][i] for age in age_groups]
    positions = [pos + i * bar_width for pos in x]
    plt.bar(positions, values, width=bar_width, label=lang.capitalize(), color=colors[i % len(colors)])

plt.xticks([j + 1.5 * bar_width for j in x], age_groups)
plt.xlabel("Age Groups")
plt.ylabel("Character Error Rate (CER)")
plt.title("Character Error Rate by Age Group and Language - Whisper")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "age_cer.png")
plt.show()

