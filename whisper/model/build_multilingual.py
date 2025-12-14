# ---------------------------------------------------------
# FIXED BUILD MULTILINGUAL SHARDS THAT MATCH preprocess.py
# ---------------------------------------------------------

import os
import logging
from pathlib import Path
from datasets import load_from_disk, concatenate_datasets, DatasetDict, Features, Sequence, Value, Audio
import argparse

BASE_PROCESSED_DIR = Path(
    "/content/drive/MyDrive/Model-Finetuning/whisper/model/processed_arrow"
)
OUTPUT_ROOT = BASE_PROCESSED_DIR.parent / "processed_arrow_multilingual"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

LANG_MAP = {
    "zu": "zulu",
    "xh": "xhosa",
    "ss": "siswati",
    "nr": "ndebele",
}

TARGET_SAMPLES = 50_000
SEED = 42

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("build_multilingual")


# --------------------------
# Load monolingual shards
# --------------------------
def load_arrow(path: Path):
    ds = load_from_disk(str(path))
    REQUIRED = {"audio", "input_features", "labels", "text"}
    assert REQUIRED.issubset(ds.column_names)
    return ds


def sample(ds, n, seed):
    if len(ds) < n:
        n = len(ds)
    return ds.shuffle(seed=seed).select(range(n))


# --------------------------
# Build multilingual
# --------------------------
def build_multilingual(langs):

    langs = list(dict.fromkeys([l.lower() for l in langs]))
    logger.info(f"Building multilingual → {langs}")

    mono = {}
    for l in langs:
        name = LANG_MAP[l]
        path = BASE_PROCESSED_DIR / name
        logger.info(f"Loading {l} → {path}")
        mono[l] = load_arrow(path)

    # Determine balanced sampling
    k = len(langs)
    base_n = TARGET_SAMPLES // k
    rem = TARGET_SAMPLES % k

    sampled = []
    for i, l in enumerate(langs):
        take = base_n + (1 if i < rem else 0)
        logger.info(f"Sampling {take} from {l}")
        sampled.append(sample(mono[l], take, seed=SEED + i))

    # Merge
    merged = concatenate_datasets(sampled)
    merged = merged.shuffle(seed=SEED)

    # ---------------------------------------------
    # ⬅️ THE IMPORTANT PART: FORCE ORIGINAL SCHEMA
    # ---------------------------------------------
    # Extract correct schema from first monolingual dataset
    example_lang = langs[0]
    template_ds = mono[example_lang]

    correct_features = Features({
        "audio": Audio(sampling_rate=16000),
        "text": Value("string"),
        "input_features": Sequence(
            Sequence(Value("float32"))
        ),
        "labels": Sequence(Value("int64")),
    })

    logger.info("Casting multilingual dataset to original schema…")
    merged = merged.cast(correct_features)

    # ---------------------------------------------
    # SAVE
    # ---------------------------------------------
    combo_name = "nguni_multilingual_" + "_".join(langs) + "_whisper"
    out_dir = OUTPUT_ROOT / combo_name

    logger.info(f"Saving multilingual → {out_dir}")
    merged.save_to_disk(str(out_dir), num_proc=1)

    logger.info(f"Done. Final size: {len(merged)}")


# --------------------------
# CLI
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", required=True)
    args = parser.parse_args()

    build_multilingual(args.languages)
