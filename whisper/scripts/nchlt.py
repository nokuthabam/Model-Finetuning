import os, json, xml.etree.ElementTree as ET
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA = BASE_DIR / "data"
# ======= CONFIG =======
LANG_CODE   = "ssw"   # e.g. 'nbl', 'zul', 'xho', 'ssw'
LANG_SHORT  = "ss"   # short language code used in the JSON: 'nr', 'ss', 'xh', 'zu'
XML_TRAIN   = DATA / "nchlt_{}".format(LANG_CODE) / "transcriptions" / "nchlt_{}.trn.xml".format(LANG_CODE)  # or your local path
XML_TEST    = DATA / "nchlt_{}".format(LANG_CODE) / "transcriptions" / "nchlt_{}.tst.xml".format(LANG_CODE)  # or your local path


AUDIO_PREFIX = r"D:/Model-Finetuning/whisper"

OUT_JSONL = DATA / f"nchlt_{LANG_CODE}.jsonl"
OUT_JSON  = DATA / f"nchlt_{LANG_CODE}.json"

OUT_JSON = DATA / f"nchlt_{LANG_SHORT}_whisper.json"

def parse_whisper(xml_path: str, audio_prefix: str, lang_short: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for spk in root.findall("speaker"):
        spk_id = spk.attrib.get("id")
        age = spk.attrib.get("age")
        for rec in spk.findall("recording"):
            rel_audio = rec.attrib.get("audio").replace("\\", "/")
            audio_path = f"{audio_prefix}/{rel_audio}"
            transcription = (rec.findtext("orth") or "").strip()
            yield {
                "audio": audio_path,
                "language": lang_short,
                "transcription": transcription,
                "speaker_id": spk_id,
                "age": age
            }

# Combine both train and test XMLs
entries = list(parse_whisper(XML_TRAIN, AUDIO_PREFIX, LANG_SHORT)) + \
          list(parse_whisper(XML_TEST, AUDIO_PREFIX, LANG_SHORT))

# ======= WRITE OUT (no list, one line per dict) =======
with open(OUT_JSON, "w", encoding="utf-8") as f:
    for entry in entries:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")

print(f"✅ Created {OUT_JSON} with {len(entries)} entries.")