import json
from sklearn.model_selection import train_test_split
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
ZULU_DATASET_PATH = BASE_DIR / "data/nchlt_zul.json"
XHOSA_DATASET_PATH = BASE_DIR / "data/nchlt_xho.json"
NDEBELE_DATASET_PATH = BASE_DIR / "data/nchlt_nbl.json"
SISWATI_DATASET_PATH = BASE_DIR / "data/nchlt_ssw.json"
# Load Lwazi entries
with open(ZULU_DATASET_PATH, "r") as f:
    entries = [json.loads(line) for line in f]

# Split (e.g., 90% train, 10% test)
train_entries, test_entries = train_test_split(entries, test_size=0.1, random_state=42)

# Save them
with open(BASE_DIR/"data/nchlt_zulu_train.json", "w") as f:
    for entry in train_entries:
        f.write(json.dumps(entry) + "\n")

with open(BASE_DIR/"data/nchlt_zulu_test.json", "w") as f:
    for entry in test_entries:
        f.write(json.dumps(entry) + "\n")


# Xhosa dataset
with open(XHOSA_DATASET_PATH, "r") as f:
    entries = [json.loads(line) for line in f] 

# split (e.g., 90% train, 10% test)
train_entries, test_entries = train_test_split(entries, test_size=0.1, random_state=42)

# Save them
with open(BASE_DIR/"data/nchlt_xhosa_train.json", "w") as f:
    for entry in train_entries:
        f.write(json.dumps(entry) + "\n")

with open(BASE_DIR/"data/nchlt_xhosa_test.json", "w") as f:
    for entry in test_entries:
        f.write(json.dumps(entry) + "\n")

# Ndebele dataset
with open(NDEBELE_DATASET_PATH, "r") as f:
    entries = [json.loads(line) for line in f]

# split (e.g., 90% train, 10% test)
train_entries, test_entries = train_test_split(entries, test_size=0.1, random_state=42)

# Save them
with open(BASE_DIR/"data/nchlt_ndebele_train.json", "w") as f:
    for entry in train_entries:
        f.write(json.dumps(entry) + "\n")
with open(BASE_DIR/"data/nchlt_ndebele_test.json", "w") as f:
    for entry in test_entries:
        f.write(json.dumps(entry) + "\n")

# Siswati dataset
with open(SISWATI_DATASET_PATH, "r") as f:
    entries = [json.loads(line) for line in f]

# split (e.g., 90% train, 10% test)
train_entries, test_entries = train_test_split(entries, test_size=0.1, random_state=42)

# Save them
with open(BASE_DIR/"data/nchlt_siswati_train.json", "w") as f:
    for entry in train_entries:
        f.write(json.dumps(entry) + "\n")

with open(BASE_DIR/"data/nchlt_siswati_test.json", "w") as f:
    for entry in test_entries:
        f.write(json.dumps(entry) + "\n")
