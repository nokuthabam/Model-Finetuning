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

# 🧼 Suppress known warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 📁 Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_NAME = "openai/whisper-small"
MODEL_DIR = BASE_DIR / "model"

# 🗂️ Setup logging
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


# 🧱 Whisper collator with shape enforcement
def make_collator(processor):
    def collate(features):
        valid = []
        for f in features:
            arr = f["input_features"]
            if isinstance(arr, torch.Tensor):
                arr = arr.cpu().numpy()
            arr = numpy.array(arr, dtype=numpy.float32)

            if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[1] == 80:
                arr = arr[0]
            elif arr.ndim == 2 and arr.shape[1] == 80:
                arr = arr.T
            elif arr.ndim == 2 and arr.shape[0] != 80:
                continue

            if arr.shape[1] < 3000:
                arr = numpy.pad(arr, ((0, 0), (0, 3000 - arr.shape[1])), mode="constant")
            elif arr.shape[1] > 3000:
                arr = arr[:, :3000]

            valid.append(torch.tensor(arr, dtype=torch.float32))

        if not valid:
            raise ValueError("No valid features")

        input_feats = torch.stack(valid)

        if "labels" in features[0]:
            labels = processor.tokenizer.pad(
                [{"input_ids": f["labels"]} for f in features],
                return_tensors="pt",
                padding=True
            ).input_ids
        else:
            texts = [f.get("text", "") for f in features]
            labels = processor.tokenizer(texts, padding=True, return_tensors="pt").input_ids

        labels[labels == processor.tokenizer.pad_token_id] = -100
        return {"input_features": input_feats, "labels": labels}
    return collate


# 🧠 Train function
def train_model(args, logger):
    lang_map = {"zu": "zulu", "xh": "xhosa", "ss": "siswati", "nr": "ndebele"}
    lang = lang_map.get(args.language, args.language)
    base_lang = "sw"  # proxy language

    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=base_lang, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.update({
        "forced_decoder_ids": processor.get_decoder_prompt_ids(language=base_lang, task="transcribe"),
        "use_timestamps": False,
        "return_timestamps": False,
        "use_cache": False,
        "suppress_tokens": []
    })
    model.generation_config.update({
        "forced_decoder_ids": model.config.forced_decoder_ids,
        "language": base_lang,
        "task": "transcribe",
        "use_timestamps": False,
        "return_timestamps": False
    })

    logger.info(f"📥 Loading dataset for {lang}")
    dataset_path = MODEL_DIR / f"processed_arrow/{lang}"
    dataset = load_from_disk(dataset_path)
    logger.info(f"✅ Loaded {len(dataset)} samples from disk")

    if not isinstance(dataset, DatasetDict):
        dataset = dataset.train_test_split(test_size=0.1, seed=42)

    dataset = dataset.shuffle(seed=42)
    dataset["test"] = dataset["test"].select(range(min(100, len(dataset["test"]))))

    logger.info(f"🧪 Train: {len(dataset['train'])} | Eval: {len(dataset['test'])}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(MODEL_DIR / f"{args.language}_whisper"),
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=1,
        learning_rate=1e-5,
        max_steps=5000,
        warmup_steps=500,
        weight_decay=0.01,
        fp16=True,
        logging_steps=200,
        save_steps=1000,
        eval_steps=1000,
        save_total_limit=2,
        evaluation_strategy="steps",
        save_strategy="steps",
        report_to="none",
        predict_with_generate=True,
        generation_max_length=150,
        load_best_model_at_end=True,
        metric_for_best_model="wer"
    )

    metric = evaluate.load("wer")

    def compute_metrics(eval_preds):
        pred_ids = eval_preds.predictions
        label_ids = eval_preds.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        preds = processor.batch_decode(pred_ids, skip_special_tokens=True)
        refs = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * metric.compute(predictions=preds, references=refs)}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=processor,
        data_collator=make_collator(processor),
        compute_metrics=compute_metrics
    )

    # Checkpoint resume logic
    ckpts = list(Path(training_args.output_dir).glob("checkpoint-*"))
    resume_ckpt = sorted(ckpts, key=lambda x: int(x.name.split("-")[1]))[-1] if ckpts else None
    if resume_ckpt:
        logger.info(f"⏮️ Resuming from {resume_ckpt}")
        for rng in Path(resume_ckpt).glob("rng_state*.pth"):
            try:
                rng.unlink()
            except:
                pass

    logger.info("🚀 Training started...")
    trainer.train(resume_from_checkpoint=str(resume_ckpt) if resume_ckpt else None)
    logger.info("✅ Training complete.")

    logger.info("💾 Saving model...")
    trainer.model.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    logger.info("📊 Running evaluation...")
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
    logger.info(f"🔥 Starting ASR training for language: {args.language}")
    train_model(args, logger)
