import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_DIR = BASE_DIR / "manifests"

df = pd.read_csv(MANIFEST_DIR / "zulu_train.csv")
print("Total rows in zulu_train.csv:", len(df))
print("Missing values:")
print(df.isnull().sum())

# csv_path = MANIFEST_DIR / "zulu_train.csv"
# df["ID"] = [f"zulu_{i:05d}" for i in range(len(df))]
# df.to_csv(csv_path, index=False)
assert df["ID"].is_unique, "❌ Duplicate IDs found!"