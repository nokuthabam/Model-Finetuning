import json
from pathlib import Path
import matplotlib.pyplot as plt
from jiwer import wer, cer

BASE_DIR = Path(__file__).resolve().parent.parent
JSON_DIR = BASE_DIR / "error_analysis"
OUTPUT_DIR = BASE_DIR / "results" / "error_analysis"
LANGUAGES = ["zulu", "xhosa", "siswati", "ndebele"]
genders = ["male", "female"]

wer_results = {lang: {} for lang in LANGUAGES}
cer_results = {lang: {} for lang in LANGUAGES}

for lang in LANGUAGES:
    for gender in genders:
        file_path = JSON_DIR / f"{lang}_{gender}.json"
        if not file_path.exists():
            print(f"File {file_path} does not exist. Skipping.")
            continue
        with open(file_path, 'r') as f:
            references, hypotheses = [], []
            for line in f:
                item = json.loads(line)
                ref = item['reference']
                hyp = item['hypothesis']
                references.append(ref)
                hypotheses.append(hyp)
            wer_results[lang][gender] = wer(references, hypotheses)
            cer_results[lang][gender] = cer(references, hypotheses)

# Plotting WER results
langs = list(wer_results.keys())
bar_width = 0.35
index = range(len(langs))
male_wer = [wer_results[lang]["male"] for lang in langs]
female_wer = [wer_results[lang]["female"] for lang in langs]

# Define colours
orange_light = "#FFA500"
orange_dark = "#FF8C00"
plt.figure(figsize=(10, 6))
plt.bar(index, male_wer, bar_width, label="Male", color=orange_light)
plt.bar([i + bar_width for i in index], female_wer, bar_width, label="Female", color=orange_dark)
plt.xlabel("Languages")
plt.ylabel("Word Error Rate (WER)")
plt.title("Word Error Rate by Gender and Language - WAV2VEC2")
plt.xticks([i + bar_width / 2 for i in index], langs)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "wer_gender_results.png")
plt.show()


# Plotting CER results
male_cer = [cer_results[lang]["male"] for lang in langs]
female_cer = [cer_results[lang]["female"] for lang in langs]

plt.figure(figsize=(10, 6))
plt.bar(index, male_cer, bar_width, label="Male", color=orange_light)
plt.bar([i + bar_width for i in index], female_cer, bar_width, label="Female", color=orange_dark)
plt.xlabel("Languages")
plt.ylabel("Character Error Rate (CER)")
plt.title("Character Error Rate by Gender and Language - WAV2VEC2")
plt.xticks([i + bar_width / 2 for i in index], langs)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cer_gender_results.png")
plt.show()