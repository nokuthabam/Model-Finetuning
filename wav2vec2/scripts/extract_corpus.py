import json
from pathlib import Path

dataset_files = [
    "commonvoice_xh_train.json",
    "commonvoice_zu_train.json",
    "nchlt_nbl.json",
    "nchlt_ssw.json",
    "nchlt_xho.json",
    "nchlt_zul.json",
    "ndebele_dataset.json",
    "siswati_dataset.json",
    "xhosa_dataset.json",
    "zulu_dataset.json",
]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "data"
count = 0
with open(OUTPUT_DIR / "combined_corpus.txt", "w", encoding="utf-8") as outfile:
    for file_name in dataset_files:
        file_path = DATA_DIR / file_name
        with open(file_path, "r", encoding="utf-8") as infile:
            
            for line in infile:
                count += 1
                try:
                    data = json.loads(line)
                    if "transcript" in data:
                        text = data["transcript"].strip().lower()
                        if text:
                            outfile.write(text + "\n")
                except json.JSONDecodeError:
                    print(f"Error decoding JSON line in {file_name}: {line}")
print("Combined corpus extraction completed.")
print(f"Total lines written to combined_corpus.txt: {count}")