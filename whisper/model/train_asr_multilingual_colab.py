import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import logging

import numpy as np
import torch
from datasets import load_from_disk, DatasetDict
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
import evaluate
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# ----------------------- CONFIG -----------------------

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_NAME = "openai/whisper-small"

# Monolingual + multilingual paths
MONO_DIR = BASE_DIR / "model" / "processed_arrow"
MULTI_DIR = BASE_DIR / "model" / "processed_arrow_multilingual"


# ----------------------- LOGGER -----------------------

def setup_logging(name):
    log_path = BASE_DIR / "logs"
    log_path.mkdir(exist_ok=True)
    log_file = log_path / f"train_{name}_{datetime.now():%Y%m%d_%H%M%S}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(name)


# ----------------------- COLLATOR -----------------------

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features):
        mel_list = []
        for f in features:
            feat = f["input_features"]
            if isinstance(feat, torch.Tensor):
                feat = feat.cpu().numpy()
            feat = np.asarray(feat, dtype=np.float32)

            if feat.ndim == 3 and feat.shape[0] == 1:
                feat = feat[0]
            mel_list.append(feat)

        max_len = max(m.shape[1] for m in mel_list)
        target_len = max(3000, max_len)

        padded = []
        for m in mel_list:
            T = m.shape[1]
            if T < target_len:
                m = np.pad(m, ((0, 0), (0, target_len - T)), mode="constant")
            else:
                m = m[:, :target_len]
            padded.append(m)

        input_features = torch.tensor(padded, dtype=torch.float32)

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        return {"input_features": input_features, "labels": labels}


def find_multilingual_dataset_dir(requested_langs, base_dir):
    """
    Finds the dataset directory whose name contains ALL requested languages,
    regardless of order. Example:
    requested_langs = ["zu", "xh"]
    matches: nguni_multilingual_zu_xh_whisper
    """

    requested_set = set(requested_langs)

    for d in base_dir.iterdir():
        if not d.is_dir():
            continue

        name = d.name

        # Must begin with multilingual prefix
        if not name.startswith("nguni_multilingual_") or not name.endswith("_whisper"):
            continue

        # Extract langs from: nguni_multilingual_zu_xh_ss_whisper
        inner = name[len("nguni_multilingual_") : -len("_whisper")]
        folder_langs = inner.split("_")

        if set(folder_langs) == requested_set:
            return d

    raise FileNotFoundError(
        f"No multilingual dataset found for languages {requested_langs} in {base_dir}"
    )


# ----------------------- TRAINER -----------------------

def train_model(args, logger):
    langs = args.languages
    combo_name = f"nguni_multilingual_{'_'.join(langs)}_whisper"
    dataset_path = find_multilingual_dataset_dir(langs, MULTI_DIR)
    logger.info(f"Matched dataset directory for languages {langs}: {dataset_path.name}")

    logger.info(f"Loading multilingual dataset: {combo_name}")
    logger.info(f"Path: {dataset_path}")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")

    dataset = load_from_disk(str(dataset_path))
    logger.info(f"Loaded {len(dataset)} multilingual samples.")

    # Split manually for multilingual
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    dataset["test"] = dataset["test"].select(range(min(400, len(dataset["test"]))))

    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME, language="en", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="en", task="transcribe"
    )
    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.language = "swahili"
    model.generation_config.task = "transcribe"
    model.generation_config.return_timestamps = False

    output_dir = BASE_DIR / f"model/{combo_name}"
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Model save directory: {output_dir}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=1,
        learning_rate=3e-5,
        warmup_steps=300,
        max_steps=1500,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=300,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        fp16=True,
        logging_steps=100,
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
    )

    metric = evaluate.load("wer")

    def compute_metrics(eval_preds):
        pred_ids = eval_preds.predictions
        label_ids = eval_preds.label_ids

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_ids = np.where(pred_ids == -100, processor.tokenizer.pad_token_id, pred_ids)

        preds = processor.batch_decode(pred_ids, skip_special_tokens=True)
        refs = processor.batch_decode(label_ids, skip_special_tokens=True)

        return {"wer": 100 * metric.compute(predictions=preds, references=refs)}

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=processor,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    # ---------------------------------------
    # 6. SAFE CHECKPOINT RESUME FIX
    # ---------------------------------------
    latest_checkpoint = None
    output_path = Path(output_dir)

    if output_path.exists():
        candidate_ckpts = list(output_path.glob("checkpoint-*"))

        if candidate_ckpts:
            latest_checkpoint = str(
                sorted(candidate_ckpts, key=lambda x: int(x.name.split("-")[1]))[-1]
            )
            logger.info(f"Resuming from checkpoint: {latest_checkpoint}")

            # DELETE RNG STATE FILES BEFORE TRAINER TOUCHES THEM
            rng_files = list(Path(latest_checkpoint).glob("rng_state*.pth")) + \
                        list(Path(latest_checkpoint).glob("pytorch_model.bin-rng_state*"))

            for f in rng_files:
                try:
                    logger.info(f"Deleting RNG state file: {f}")
                    f.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete {f}: {e}")

        else:
            logger.info("No checkpoints found. Training from scratch.")
    else:
        logger.info("Output directory does not exist yet. Training from scratch.")


    logger.info("Training started...")
    trainer.train(resume_from_checkpoint=str(latest_checkpoint) if latest_checkpoint else None)
    logger.info("Training complete!")

    trainer.model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    final_metrics = trainer.evaluate()
    logger.info(f"Final WER: {final_metrics}")

    results_dir = BASE_DIR / "results" / combo_name
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    logger.info(f"Saved results to {results_dir}")


# ----------------------- MAIN -----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--languages",
        nargs="+",
        required=True,
        help="Specify languages, e.g. --languages zu xh ss nr"
    )
    args = parser.parse_args()

    name = "_".join(args.languages)
    logger = setup_logging(name)
    train_model(args, logger)
