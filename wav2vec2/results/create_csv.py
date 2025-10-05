import json
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"

files = [
    RESULTS / "zulu_analysis_results.json",
    RESULTS / "xhosa_analysis_results.json",
    RESULTS / "ndebele_analysis_results.json",
    RESULTS / "siswati_analysis_results.json",
]

lang_codes = ["zu", "xh", "nbl", "ssw"]
name_to_code = {
    "zulu": "zu",
    "xhosa": "xh",
    "ndebele": "nbl",
    "siswati": "ssw",
    }

def parse_entry(source_lang_code = str, model_key = str, metrics = dict):
    key = model_key.lower()
    model_type = "multilingual" if "multilingual" in key else "monolingual"
    langs_in_key = []
    if model_type == "monolingual":
        langs_in_key = [source_lang_code]
    else:
        for token in key.split("_"):
            if token in lang_codes and token not in langs_in_key:
                langs_in_key.append(token)
            
        if not langs_in_key:
            langs_in_key = [source_lang_code]
    
    return {
        "model_type": model_type,
        "languages": ",".join(langs_in_key),
        "wer": float(metrics.get("wer")) if "wer" in metrics else None,
        "cer": float(metrics.get("cer")) if "cer" in metrics else None,
        "model_key": model_key
    }


rows =[]
for f in files:
    if not f.exists():
        continue
    first_token = f.stem.split("_")[0].lower()
    source_code = name_to_code.get(first_token, first_token)
    with open(f, "r", encoding="utf-8") as infile:
        data = json.load(infile)
        for k, v in data.items():
            if isinstance(v, dict) and {"wer", "cer"} <= v.keys():
                rows.append(parse_entry(source_code, k, v))

df = pd.DataFrame(rows, columns = ["model_type", "languages", "wer", "cer", "model_key"])
df.to_csv(RESULTS / "wav2vec2_analysis_summary.csv", index=False)

for code in sorted(lang_codes):
    sub = df[df["languages"].str.contains(fr"\b{code}\b")]
    if not sub.empty:
        sub.to_csv(RESULTS / f"wav2vec2_analysis_summary_{code}.csv", index=False)