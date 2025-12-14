from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
# Point to your JSONs
json_files = [
    DATA_DIR / "nchlt_ndebele_test.json",
    DATA_DIR / "nchlt_xhosa_test.json",
    DATA_DIR / "nchlt_zulu_test.json",
    DATA_DIR / "nchlt_siswati_test.json",
]


def fix_audio_paths(entry):
    audio_path = entry.get("audio_path", "")
    if "wav2vec2/data" not in audio_path:
        entry["audio_path"] = audio_path.replace("wav2vec2", "wav2vec2\\data", 1)
        return True
    return False


for file_path in json_files:
    modified = False
    data = []

    try:
        # Try to load as standard JSON
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        file_type = "json"
    except json.JSONDecodeError:
        # Fallback: treat as JSONL
        file_type = "jsonl"
        with open(file_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]

    # Fix paths
    for entry in data:
        if fix_audio_paths(entry):
            modified = True

    # Write new file if modified
    if modified:
        output_path = file_path.with_name(file_path.stem + "_fixedpaths" + file_path.suffix)
        with open(output_path, "w", encoding="utf-8") as f:
            if file_type == "json":
                json.dump(data, f, ensure_ascii=False, indent=2)
            else:  # jsonl
                for entry in data:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"✅ Fixed and saved: {output_path}")
    else:
        print(f"✅ No changes needed for: {file_path}")
