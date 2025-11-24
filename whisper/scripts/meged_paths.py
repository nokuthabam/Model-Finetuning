import json
from pathlib import Path

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
# merged JSONs to fix
MERGED_FILES = [
    DATA_DIR / "merged_zu.json",
    DATA_DIR / "merged_xh.json",
    DATA_DIR / "merged_nr.json",
    DATA_DIR / "merged_ss.json",
]

# Windows base path in your JSONs
WINDOWS_PREFIX = "D:/Model-Finetuning"
# New prefix for Colab path
COLAB_PREFIX = "/content/drive/MyDrive/Model-Finetuning"


# -------------------------------------------------------------
# FUNCTION
# -------------------------------------------------------------
def fix_paths(json_path):
    print(f"\n🔧 Fixing paths in: {json_path.name}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    count = 0
    for entry in data:
        if "audio" in entry:
            old = entry["audio"]
            # unify slashes
            old_clean = old.replace("\\", "/")

            # replace drive prefix
            new = old_clean.replace(WINDOWS_PREFIX, COLAB_PREFIX)

            if new != entry["audio"]:
                entry["audio"] = new
                count += 1

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✔ Updated {count} paths in {json_path.name}")


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
if __name__ == "__main__":
    for fpath in MERGED_FILES:
        if fpath.exists():
            fix_paths(fpath)
        else:
            print(f"⚠️ Missing: {fpath}")
