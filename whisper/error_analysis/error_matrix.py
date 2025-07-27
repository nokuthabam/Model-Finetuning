import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "error_analysis"

REPORT_FILES = [
    "zulu_error_analysis_report.json",
    "xhosa_error_analysis_report.json",
    "siswati_error_analysis_report.json",
    "ndebele_error_analysis_report.json"
]

def load_substitutions(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    lang = data["language"]
    subs = data["phoneme_substitutions"]
    counts = defaultdict(int)
    for item in subs:
        try:
            source, target = item["substitution"].split(" →")
            counts[(source, target)] += item["count"]
            print(f"Loaded substitution: {source} → {target} with count {item['count']}")
        except ValueError:
            continue
    return lang, counts

# --- Build Error Matrix ---
all_pairs = set()
lang_counts = {}

for file in REPORT_FILES:
    lang, counts = load_substitutions(RESULTS_DIR / file)
    print(f"Loaded {len(counts)} substitutions for {lang}")
    lang_counts[lang] = counts
    all_pairs.update(counts.keys())

# Organize data into DataFrame
df = pd.DataFrame(index=sorted(set(source for source, _ in all_pairs)))

for lang, counts in lang_counts.items():
    col_data = defaultdict(int)
    for (source, target), count in counts.items():
        key = f"{source} → {target}"
        col_data[key] = count
    df = df.join(pd.Series(col_data, name=lang), how='outer')

df.fillna(0, inplace=True)
df = df.astype(int).sort_index()
# --- Plot Heat Maps ---
for lang in lang_counts:
    plt.figure(figsize=(10, 6))
    lang_df = df[lang]
    lang_matrix = lang_df[lang_df > 0].sort_values(ascending=False)[:30]
    pairs = [s.split(" → ") for s in lang_matrix.index]
    sub_df = pd.DataFrame({
        "From": [source for source, target in pairs],
        "To": [target for source, target in pairs],
        "Count": lang_matrix.values
    })

    

    heatmap_data = sub_df.pivot(index='From', columns='To', values='Count')
    if heatmap_data.size == 0:
        print(f"No substitution data to plot for {lang}")
        continue
    sns.heatmap(heatmap_data, annot=True, fmt="g", cmap="Reds", linewidths=0.5)
    plt.title(f"Top Character Substitutions for {lang.capitalize()}")
    plt.xlabel("Substituted With")
    plt.ylabel("Reference Character")
    plt.tight_layout()
    plt.show()