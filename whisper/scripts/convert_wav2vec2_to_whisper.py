import json
from pathlib import Path

# ISO 639-1 codes
LANGUAGE_CODES = {
    "zulu": "zu",
    "xhosa": "xh",
    "ndebele": "nr",
    "siswati": "ss"
}

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

json_files = list(DATA_DIR.glob("*.json"))

def convert_file(json_path):
    name_parts = json_path.stem.split("_") # e.g., "zulu_train"
    base_lang = name_parts[0] # e.g., "zulu"

    # Map to ISO 639-1 code
    iso_code = LANGUAGE_CODES.get(base_lang, base_lang)
    if not iso_code:
        print(f"⚠️ Unsupported language: {base_lang}")
        return
    
    updated_lines = []
    with json_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            # Standardize audio path for Whisper
            whisper_item = {
                "audio": item["audio_path"].replace("\\", "/"),
                "language": iso_code,
                "transcription": item["transcript"],
                "speaker_id": item.get("speaker_id", "unknown"),
                "age": str(item.get("age", "")),  # Ensure age is a string
            }


            updated_lines.append(json.dumps(whisper_item, ensure_ascii=False))

        with open(json_path, "w", encoding="utf-8") as out_f:
            out_f.write("\n".join(updated_lines))
        print(f"✅ Converted {json_path.name} to Whisper format")

for json_file in json_files:
    # check if the file contains either language name or language code
    if any(lang in json_file.stem for lang in LANGUAGE_CODES.keys()) or any(code in json_file.stem for code in LANGUAGE_CODES.values()):
        
        print(f"🔄 Processing: {json_file.name}")
        convert_file(json_file)
    else:
        print(f"⚠️ Skipping unsupported file: {json_file.name}")
