import json
from pathlib import Path

def fix_age_field_in_place(file_path: Path):
    print(f"🔧 Fixing: {file_path.name}")
    fixed_lines = []
    with file_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                if "age" in obj and not isinstance(obj["age"], str):
                    obj["age"] = str(obj["age"])
                fixed_lines.append(json.dumps(obj, ensure_ascii=False))
            except json.JSONDecodeError:
                print(f"⚠️ Skipping malformed line {i}")
    
    with file_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(fixed_lines))
    print(f"✅ Fixed {len(fixed_lines)} entries\n")

# 🗂 Update these as needed
json_files = [
    Path("D:/Model-Finetuning/wav2vec2/data/ndebele_train.json"),
    Path("D:/Model-Finetuning/wav2vec2/data/ndebele_test.json"),
    Path("D:/Model-Finetuning/wav2vec2/data/xhosa_train.json"),
    Path("D:/Model-Finetuning/wav2vec2/data/xhosa_test.json"),
    Path("D:/Model-Finetuning/wav2vec2/data/siswati_train.json"),
    Path("D:/Model-Finetuning/wav2vec2/data/siswati_test.json"),
    Path("D:/Model-Finetuning/wav2vec2/data/zulu_train.json"),
    Path("D:/Model-Finetuning/wav2vec2/data/zulu_test.json"),
]

for file in json_files:
    fix_age_field_in_place(file)
