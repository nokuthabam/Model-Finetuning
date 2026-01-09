# from pathlib import Path
# import json

# BASE_DIR = Path(__file__).resolve().parent.parent
# DATA_DIR = BASE_DIR / "data"
# # Point to your JSONs
# json_files = [
#     DATA_DIR / "nchlt_zulu_test.json",
#     DATA_DIR / "nchlt_xhosa_test.json",
#     DATA_DIR / "nchlt_ndebele_test.json",
#     DATA_DIR / "nchlt_siswati_test.json",
# ]


# def fix_audio_paths(entry):
#     audio_path = entry.get("audio", "")
#     if "whisper/data" not in audio_path:
#         entry["audio"] = audio_path.replace("whisper", "whisper/data", 1)
#         return True
#     return False


# for file_path in json_files:
#     modified = False
#     data = []

#     try:
#         # Try to load as standard JSON
#         with open(file_path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         file_type = "json"
#     except json.JSONDecodeError:
#         # Fallback: treat as JSONL
#         file_type = "jsonl"
#         with open(file_path, "r", encoding="utf-8") as f:
#             data = [json.loads(line) for line in f if line.strip()]

#     # Fix paths
#     for entry in data:
#         if fix_audio_paths(entry):
#             modified = True

#     # Write new file if modified
#     if modified:
#         output_path = file_path.with_name(file_path.stem + "_fixedpaths" + file_path.suffix)
#         with open(output_path, "w", encoding="utf-8") as f:
#             if file_type == "json":
#                 json.dump(data, f, ensure_ascii=False, indent=2)
#             else:  # jsonl
#                 for entry in data:
#                     f.write(json.dumps(entry, ensure_ascii=False) + "\n")
#         print(f"✅ Fixed and saved: {output_path}")
#     else:
#         print(f"✅ No changes needed for: {file_path}")


import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ERROR_ANALYSIS_DIR = BASE_DIR / "error_analysis"
in_path = ERROR_ANALYSIS_DIR / "xhosa_inference_results.json"      # change if needed
out_path = ERROR_ANALYSIS_DIR / "xhosa_inference_results_fixed.json"

OLD = "/wav2vec2/data/"
NEW = "/whisper/data/"

changed = 0

with in_path.open("r", encoding="utf-8") as fin, \
     out_path.open("w", encoding="utf-8") as fout:

    for line in fin:
        line = line.strip()
        if not line:
            continue

        obj = json.loads(line)

        audio = obj.get("audio", "")
        if OLD in audio:
            obj["audio"] = audio.replace(OLD, NEW)
            changed += 1

        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Done. Updated {changed} paths → {out_path}")