import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MERGED_FILES = [
    BASE_DIR / "merged_zu.json",
    BASE_DIR / "merged_xh.json",
    BASE_DIR / "merged_nr.json",
    BASE_DIR / "merged_ss.json",
]

def fix_paths_in_json(json_path):
    print(f"Fixing paths in: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        if "audio" in entry:
            entry["audio"] = entry["audio"].replace("\\", "/")

    # save back
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✔ Updated: {json_path}")


# MAIN LOOP
for file in MERGED_FILES:
    if file.exists():
        fix_paths_in_json(file)
    else:
        print(f"⚠️ File not found: {file}")
