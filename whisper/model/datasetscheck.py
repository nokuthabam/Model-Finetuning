from datasets import load_from_disk

import pprint 
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SAVE_DIR = BASE_DIR / "model" / "processed_arrow"

ds = load_from_disk(SAVE_DIR / "ndebele")
pprint.pprint(ds.features)

# print(ds[0])
#print({k: type(v) for k, v in ds[0].items()})
lens = []
for i in range(5):  # check first 20 safely
    audio = ds["audio"][i]
    lens.append(len(audio["array"]))

print(lens)

print("min:", min(lens), "max:", max(lens), "avg:", sum(lens)/len(lens))