import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

LANGUAGE_MAP = {
    "zu": "zulu",
    "xh": "xhosa",
    "ssw": "siswati",
    "nbl": "ndebele"
}

def create_unseen_dataset(language_code):
    """
    Create a dataset for unseen data
    """
    language = LANGUAGE_MAP.get(language_code)
    train_path = DATA_DIR / f"{LANGUAGE_MAP[language_code]}_train.json"
    test_path = DATA_DIR / f"{LANGUAGE_MAP[language_code]}_test.json"
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Dataset files not found for language code: {language_code}")
    
    with open(train_path, 'r') as f:
        train_lines = [json.loads(line) for line in f.readlines()[1001:]]
    
    with open(test_path, 'r') as f:
        test_lines = [json.loads(line) for line in f.readlines()[101:]]

    unseen_dataset = train_lines + test_lines

    with open(DATA_DIR / f"{language}_unseen.json", 'w') as f:
        for item in unseen_dataset:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    for lang_code in LANGUAGE_MAP.keys():
        create_unseen_dataset(lang_code)
        print(f"Unseen dataset created for {LANGUAGE_MAP[lang_code]}.")