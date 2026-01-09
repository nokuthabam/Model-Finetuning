import os, json, xml.etree.ElementTree as ET
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA = BASE_DIR / "data"
# ======= CONFIG =======
LANG_CODE   = "zul"   # e.g. 'nbl', 'zul', 'xho', 'ssw'
XML_TRAIN   = DATA / "nchlt_{}".format(LANG_CODE) / "transcriptions" / "nchlt_{}.trn.xml".format(LANG_CODE)  # or your local path
XML_TEST    = DATA / "nchlt_{}".format(LANG_CODE) / "transcriptions" / "nchlt_{}.tst.xml".format(LANG_CODE)  # or your local path


AUDIO_PREFIX = r"D:\\Model-Finetuning\\wav2vec2\\data"

OUT_JSONL = DATA / f"nchlt_{LANG_CODE}.jsonl"
OUT_JSON  = DATA / f"nchlt_{LANG_CODE}.json"

# ======= PARSER =======
def parse_min(xml_path: str, audio_prefix: str):
    rows = []
    root = ET.parse(xml_path).getroot()
    for spk in root.findall("speaker"):
        spk_id = spk.attrib.get("id")
        age = spk.attrib.get("age")
        gender = spk.attrib.get("gender")
        for rec in spk.findall("recording"):
            rel_audio = rec.attrib.get("audio")  # e.g. 'nchlt_nbl/audio/001/nchlt_nbl_001m_0007.wav'
            audio_path = str(Path(audio_prefix) / Path(rel_audio.replace("/", os.sep)))
            transcript = (rec.findtext("orth") or "").strip()
            rows.append({
                "audio_path": audio_path,
                "transcript": transcript,
                "speaker_id": spk_id,
                "age": age,
                "gender": gender
            })
    return rows

rows = parse_min(XML_TRAIN, AUDIO_PREFIX) + parse_min(XML_TEST, AUDIO_PREFIX)

with open(OUT_JSON, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"created {OUT_JSON} with {len(rows)} items")