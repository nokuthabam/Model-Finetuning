import os, json, torch, logging, re, argparse, warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Union

from datasets import load_dataset, concatenate_datasets, DatasetDict, Value
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    TrainingArguments,
    Trainer,
)
import torchaudio
import evaluate

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ================================
# Paths (works in Windows + Colab)
# ================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results"
LOG_DIR = BASE_DIR / "logs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ======================
# Model + Vocab
# ======================
MODEL_NAME = "facebook/wav2vec2-xls-r-300m"

# Custom vocab dictionary
vocab_dict = {
    "|": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8, "i": 9,
    "j": 10, "k": 11, "l": 12, "m": 13, "n": 14, "o": 15, "p": 16, "q": 17, "r": 18,
    "s": 19, "t": 20, "u": 21, "v": 22, "w": 23, "x": 24, "y": 25, "z": 26,
    "[UNK]": 27, "[PAD]": 28
}

with open(MODEL_DIR / "vocab.json", "w") as f:
    json.dump(vocab_dict, f)

tokenizer = Wav2Vec2CTCTokenizer(
    vocab_file=str(MODEL_DIR / "vocab.json"),
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|"
)

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)

processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer
)

# ======================
# Metrics
# ======================
wer_metric = evaluate.load("wer")
CER_REMOVE = r'[\,\?\.\!\-\;\:\"\“\%\‘\”\'\…\•\°\(\)\=\*\/\`\ː\’]'

# ======================
# Logging
# ======================
def setup_logging(lang_list):
    name = "_".join(lang_list)
    log_file = LOG_DIR / f"train_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


# ======================
# Text Cleaning
# ======================
def clean_text(text):
    text = text.lower().strip()
    text = text.replace("[n]", "").replace("[s]", "").replace("[um]", "")
    text = text.replace("[", "").replace("]", "")
    text = re.sub(CER_REMOVE, "", text)
    text = re.sub(r"\s+", " ", text)
    return text


# ======================
# Audio Loading
# ======================
def speech_file_to_array_fn(batch):
    waveform, sr = torchaudio.load(batch["audio_path"])
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        sr = 16000
    batch["speech"] = waveform.squeeze().numpy()
    batch["sampling_rate"] = sr
    return batch


def prepare_dataset(batch):
    batch["transcript"] = clean_text(batch["transcript"])

    if isinstance(batch["speech"], list):
        inputs = processor(batch["speech"], sampling_rate=16000).input_values
        with processor.as_target_processor():
            labels = processor(batch["transcript"]).input_ids

        return {
            "input_values": inputs[0],
            "labels": labels,
        }

    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"]).input_values[0]
    batch["labels"] = processor(batch["transcript"]).input_ids
    return batch


# ======================
# Collator
# ======================
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features):
        input_feats = [{"input_values": f["input_values"]} for f in features]
        label_feats = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(input_feats, padding=self.padding, return_tensors="pt")

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_feats, padding=self.padding, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


# ==========================
# Metric computation
# ==========================
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# ================================
# Multilingual Data Loader
# ================================
def load_and_merge_multilingual(logger, language_codes):

    code_map = {
        "zu": "zulu",
        "xh": "xhosa",
        "ssw": "siswati",
        "nbl": "ndebele",
    }

    N = len(language_codes)
    if N == 1:
        k = 15000
    elif N == 2:
        k = 10000
    elif N == 3:
        k = 3333
    else:
        k = 5000

    logger.info(f"Sampling {k} samples per language across {N} languages.")

    all_train, all_test = [], []

    for code in language_codes:
        name = code_map[code]
        paths = {
            "lwazi_train": DATA_DIR / f"{name}_train.json",
            "lwazi_test": DATA_DIR / f"{name}_test.json",
            "cv_train": DATA_DIR / f"commonvoice_{code}_train.json",
            "nchlt_train": DATA_DIR / f"nchlt_{name}_train.json",
            "nchlt_test": DATA_DIR / f"nchlt_{name}_test.json",
        }

        logger.info(f"Loading datasets for {code}")

        lwazi_train = load_dataset("json", data_files=str(paths["lwazi_train"]))["train"]
        lwazi_test = load_dataset("json", data_files=str(paths["lwazi_test"]))["train"]

        train_parts = [lwazi_train]
        test_parts = [lwazi_test]

        for src in ["cv_train", "nchlt_train"]:
            if paths[src].exists():
                ds = load_dataset("json", data_files=str(paths[src]))["train"]
                ds = ds.cast_column("age", Value("string"))
                train_parts.append(ds)

        if paths["nchlt_test"].exists():
            ds_test = load_dataset("json", data_files=str(paths["nchlt_test"]))["train"]
            test_parts.append(ds_test)

        train_combined = concatenate_datasets(train_parts).shuffle(seed=42)
        test_combined = concatenate_datasets(test_parts).shuffle(seed=42)

        train_combined = train_combined.select(range(min(k, len(train_combined))))
        test_combined = test_combined.select(range(min(3000, len(test_combined))))

        all_train.append(train_combined)
        all_test.append(test_combined)

    merged_train = concatenate_datasets(all_train)
    merged_test = concatenate_datasets(all_test)

    return DatasetDict({"train": merged_train, "test": merged_test})


# ===========================
# Training Function
# ===========================
def train_multilingual_model(languages, logger):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset = load_and_merge_multilingual(logger, languages)

    dataset = dataset.map(speech_file_to_array_fn, num_proc=8)
    dataset = dataset.map(prepare_dataset)

    model_id = "_".join(languages)
    output_dir = MODEL_DIR / f"{model_id}_wav2vec2"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_NAME,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        gradient_checkpointing=True,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        group_by_length=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        logging_steps=500,
        save_steps=500,
        eval_steps=500,
        num_train_epochs=10,
        fp16=torch.cuda.is_available(),
        learning_rate=3e-4,
        metric_for_best_model="wer",
        greater_is_better=False,
        load_best_model_at_end=True,
        save_total_limit=2,
        warmup_steps=500,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorCTCWithPadding(processor=processor),
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=processor.feature_extractor,
    )

    logger.info("Beginning training...")
    trainer.train()

    logger.info("Saving model...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)


# ===========================
# CLI Entry
# ===========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multilingual Wav2Vec2 ASR model")
    parser.add_argument(
        "--languages",
        nargs="+",
        type=str,
        required=True,
        help="Languages to train on: zu xh ssw nbl",
    )
    args = parser.parse_args()

    logger = setup_logging(args.languages)
    logger.info(f"Training on languages: {args.languages}")

    train_multilingual_model(args.languages, logger)

    logger.info("Training complete.")
