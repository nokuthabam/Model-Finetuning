import os
import logging
from pathlib import Path

from datasets import load_from_disk, concatenate_datasets

# ------------ CONFIG ------------
BASE_PROCESSED_DIR = Path(
    "/content/drive/MyDrive/Model-Finetuning/whisper/model/processed_arrow"
)

OUTPUT_ROOT = BASE_PROCESSED_DIR.parent / "processed_arrow_multilingual"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET_TOTAL_SAMPLES = 50_000

# Map short language codes to their monolingual Arrow dirs
LANG_DIRS = {
    "zu": BASE_PROCESSED_DIR / "zulu",
    "xh": BASE_PROCESSED_DIR / "xhosa",
    "nr": BASE_PROCESSED_DIR / "ndebele",
    "ss": BASE_PROCESSED_DIR / "siswati",
}

# All multilingual combinations you showed in your screenshot
COMBOS = {
    # 2-language
    "nguni_multilingual_zu_xh_whisper": ["zu", "xh"],
    "nguni_multilingual_zu_nr_whisper": ["zu", "nr"],
    "nguni_multilingual_zu_ss_whisper": ["zu", "ss"],
    "nguni_multilingual_xh_nr_whisper": ["xh", "nr"],
    "nguni_multilingual_xh_ss_whisper": ["xh", "ss"],
    "nguni_multilingual_nr_ss_whisper": ["nr", "ss"],

    # 3-language
    "nguni_multilingual_zu_xh_nr_whisper": ["zu", "xh", "nr"],
    "nguni_multilingual_zu_xh_ss_whisper": ["zu", "xh", "ss"],
    "nguni_multilingual_zu_nr_ss_whisper": ["zu", "nr", "ss"],
    "nguni_multilingual_xh_nr_ss_whisper": ["xh", "nr", "ss"],

    # 4-language
    "nguni_multilingual_zu_xh_nr_ss_whisper": ["zu", "xh", "nr", "ss"],
}
# ---------------------------------


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("build_multilingual_arrows")


def sample_balanced(ds, n, lang, seed):
    """Shuffle and take n samples from a monolingual dataset."""
    ds_size = len(ds)
    if ds_size < n:
        logger.warning(
            f"[{lang}] Requested {n} samples but only {ds_size} available. "
            f"Using all {ds_size}."
        )
        n = ds_size
    return ds.shuffle(seed=seed).select(range(n))


def main():
    logger.info("🔧 Loading monolingual datasets from disk...")

    mono = {}
    for lang, path in LANG_DIRS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing monolingual dataset for {lang}: {path}")
        logger.info(f"Loading {lang} from {path} ...")
        mono[lang] = load_from_disk(str(path))
        logger.info(f"{lang} size: {len(mono[lang])} examples")

    logger.info("✅ Monolingual datasets loaded.")

    for combo_name, langs in COMBOS.items():
        logger.info("=" * 80)
        logger.info(f"📦 Building multilingual dataset: {combo_name}")
        logger.info(f"Languages: {', '.join(langs)}")

        k = len(langs)
        base_n = TARGET_TOTAL_SAMPLES // k
        remainder = TARGET_TOTAL_SAMPLES % k

        logger.info(
            f"Target total: {TARGET_TOTAL_SAMPLES} | "
            f"{k} languages -> base {base_n} each, remainder {remainder}"
        )

        sampled_parts = []
        for idx, lang in enumerate(langs):
            # Distribute any remainder to the first `remainder` languages
            n_for_lang = base_n + (1 if idx < remainder else 0)
            logger.info(f"Sampling {n_for_lang} examples from {lang}...")
            sampled = sample_balanced(mono[lang], n_for_lang, lang, seed=SEED + idx)
            sampled_parts.append(sampled)
            logger.info(f" -> got {len(sampled)} examples from {lang}")

        multilingual = concatenate_datasets(sampled_parts)
        multilingual = multilingual.shuffle(seed=SEED)

        logger.info(
            f"✅ Final size for {combo_name}: {len(multilingual)} examples "
            f"(requested {TARGET_TOTAL_SAMPLES})"
        )

        out_dir = OUTPUT_ROOT / combo_name
        logger.info(f"💾 Saving to {out_dir} ...")
        multilingual.save_to_disk(str(out_dir))
        logger.info(f"✅ Saved {combo_name}.\n")

    logger.info("🎉 All multilingual Arrow shards created successfully.")


if __name__ == "__main__":
    main()
