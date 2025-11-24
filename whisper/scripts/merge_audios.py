import os, json
from pydub import AudioSegment
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

MAX_DURATION = 30_000  # 30 seconds
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# all json files you want to process
JSON_FILES = {
    "zu": [DATA_DIR / "nchlt_zu_whisper.json", DATA_DIR / "zulu_dataset.json"],
    "xh": [DATA_DIR / "nchlt_xh_whisper.json", DATA_DIR / "xhosa_dataset.json"],
    "nr": [DATA_DIR / "nchlt_nr_whisper.json", DATA_DIR / "ndebele_dataset.json"],
    "ss": [DATA_DIR / "nchlt_ss_whisper.json", DATA_DIR / "siswati_dataset.json"],
}

OUTPUT_DIR = BASE_DIR / "merged_audio"
OUTPUT_DIR.mkdir(exist_ok=True)


def merge_dataset(json_path: Path, lang_code: str, lang_dir: Path):
    print(f"\nProcessing {json_path} ({lang_code})")

    # load JSONL OR JSON array
    with open(json_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        f.seek(0)

        if first_line.startswith("{"):  
            # JSONL file
            entries = [json.loads(line) for line in f if line.strip().startswith("{")]
        else:
            # Standard JSON array
            entries = json.load(f)

    # group entries by speaker
    groups = defaultdict(list)
    for e in entries:
        groups[e["speaker_id"]].append(e)

    merged_output = []

    # merge for each speaker
    for spk, samples in tqdm(groups.items(), desc=f"Merging speakers for {lang_code}"):
        samples_sorted = sorted(samples, key=lambda x: x["audio"])
        
        buffer_audio = AudioSegment.silent(duration=0)
        buffer_text = ""
        part = 1
        
        for item in samples_sorted:
            try:
                audio = AudioSegment.from_wav(item["audio"])
            except Exception as e:
                print(f"Could not load: {item['audio']} ({e})")
                continue
            
            if len(buffer_audio) + len(audio) <= MAX_DURATION:
                buffer_audio += audio
                buffer_text += " " + item["transcription"]
            else:
                # write current buffer
                out_file = lang_dir / f"{spk}_{part}.wav"
                buffer_audio.export(out_file, format="wav")
                
                merged_output.append({
                    "audio": str(out_file),
                    "language": lang_code,
                    "speaker_id": spk,
                    "transcription": buffer_text.strip()
                })
                
                part += 1
                buffer_audio = audio
                buffer_text = item["transcription"]
        
        # write last segment
        if len(buffer_audio) > 0:
            out_file = lang_dir / f"{spk}_{part}.wav"
            buffer_audio.export(out_file, format="wav")
            merged_output.append({
                "audio": str(out_file),
                "language": lang_code,
                "speaker_id": spk,
                "transcription": buffer_text.strip()
            })

    return merged_output


# MAIN LOOP
all_merged_by_lang = {lang: [] for lang in JSON_FILES.keys()}

for lang, files in JSON_FILES.items():
    lang_dir = OUTPUT_DIR / lang
    lang_dir.mkdir(exist_ok=True)

    for json_file in files:
        if Path(json_file).exists():
            merged_segments = merge_dataset(Path(json_file), lang, lang_dir)
            all_merged_by_lang[lang].extend(merged_segments)
        else:
            print(f"⚠️ File not found: {json_file}")

# write one combined JSON per language
for lang, merged_entries in all_merged_by_lang.items():
    out_json = BASE_DIR / f"merged_{lang}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(merged_entries, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved combined merged dataset → {out_json} ({len(merged_entries)} segments)")
