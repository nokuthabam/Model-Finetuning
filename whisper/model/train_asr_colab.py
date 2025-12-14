import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import logging

import numpy
import torch
from datasets import load_from_disk, DatasetDict
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    GenerationConfig,
)
import evaluate
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Suppress known warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_NAME = "openai/whisper-small"
MODEL_DIR = BASE_DIR / "model"

# Setup logging
def setup_logging(lang_code):
    log_path = BASE_DIR / "logs"
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / f"train_{lang_code}_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


# Whisper collator with shape enforcement
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        """
        Whisper-compatible collator:
        - Accepts variable-length mel features
        - Pads to max(batch_len, 3000)
        - Removes BOS
        """

        # -----------------------------
        # 1. Collect mel features
        # -----------------------------
        mel_list = []
        for f in features:
            feat = f["input_features"]

            if isinstance(feat, torch.Tensor):
                feat = feat.cpu().numpy()

            feat = numpy.asarray(feat, dtype=numpy.float32)

            # Squeeze [1, 80, T] -> [80, T]
            if feat.ndim == 3 and feat.shape[0] == 1:
                feat = feat[0]

            mel_list.append(feat)  # shape [80, T]

        # -----------------------------
        # 2. Pad mel features
        # -----------------------------
        # Whisper requires:   T == 3000
        # Our rule: pad to max(batch_T, 3000)
        max_batch_len = max(m.shape[1] for m in mel_list)
        target_len = max(3000, max_batch_len)

        padded = []
        for m in mel_list:
            T = m.shape[1]

            if T < target_len:
                m = numpy.pad(m, ((0, 0), (0, target_len - T)), mode="constant")
            else:
                m = m[:, :target_len]

            padded.append(m)

        input_features = torch.tensor(padded, dtype=torch.float32)

        # -----------------------------
        # 3. Pad labels
        # -----------------------------
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=True,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"]
        labels = labels.masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove BOS if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        return {
            "input_features": input_features,
            "labels": labels,
        }


# Train function
def train_model(args, logger):
    lang_map = {"zu": "zulu", "xh": "xhosa", "ss": "siswati", "nr": "ndebele"}
    lang = lang_map.get(args.language, args.language)
    base_lang = "en"  # proxy language

    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=base_lang, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=base_lang,
        task="transcribe",
    )
    model.config.forced_decoder_ids = forced_decoder_ids
    # model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.language = "swahili"
    model.generation_config.task = "transcribe"
    model.generation_config.return_timestamps = False
    # model.config.suppress_tokens = []

    logger.info(f"Loading dataset for {lang}")
    dataset_path = MODEL_DIR / f"processed_arrow/{lang}"
    dataset = load_from_disk(dataset_path)
    logger.info(f"Loaded {len(dataset)} samples from disk")

    if not isinstance(dataset, DatasetDict):
        dataset = dataset.train_test_split(test_size=0.1, seed=42)

    dataset = dataset.shuffle(seed=42)
    dataset["test"] = dataset["test"].select(range(min(400, len(dataset["test"]))))

    logger.info(f"Train: {len(dataset['train'])} | Eval: {len(dataset['test'])}")
    
    output_dir = BASE_DIR / f"model/{args.language}_whisper"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory set to {output_dir}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=1,
        learning_rate=3e-5,
        max_steps=1500,
        warmup_steps=300,
        weight_decay=0.0,
        fp16=True,
        logging_steps=100,
        save_steps=100,
        eval_steps=100,
        save_total_limit=2,
        evaluation_strategy="steps",
        save_strategy="steps",
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        # label_smoothing_factor=0.1,
        predict_with_generate=True,
        generation_max_length=300,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    metric = evaluate.load("wer")

    def compute_metrics(eval_preds):
        pred_ids = eval_preds.predictions
        label_ids = eval_preds.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_ids = numpy.where(pred_ids == -100, processor.tokenizer.pad_token_id, pred_ids)
        preds = processor.batch_decode(pred_ids, skip_special_tokens=True)
        refs = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * metric.compute(predictions=preds, references=refs)}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=processor,
        data_collator=data_collator,
        compute_metrics=compute_metrics
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
    logger.info("✅ Training complete.")

    logger.info("Saving model...")
    trainer.model.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    logger.info("Running evaluation...")
    results = trainer.evaluate()
    result_path = BASE_DIR / f"results/{args.language}_whisper/metrics.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"✅ Results saved to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, required=True, help="Language code: zu | xh | ss | nr")
    args = parser.parse_args()

    logger = setup_logging(args.language)
    logger.info(f"Starting ASR training for language: {args.language}")
    train_model(args, logger)
