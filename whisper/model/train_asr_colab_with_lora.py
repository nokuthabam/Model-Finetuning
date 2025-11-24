import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import logging
import warnings

import numpy
import torch
import evaluate
from datasets import load_from_disk, DatasetDict
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------------------------------------
# CONFIG & ENVIRONMENT
# ---------------------------------------
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_NAME = "openai/whisper-small"
MODEL_DIR = BASE_DIR / "model"

# ---------------------------------------
# LOGGING SETUP
# ---------------------------------------
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

# ---------------------------------------
# COLLATOR
# ---------------------------------------
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

# ---------------------------------------
# TRAIN FUNCTION
# ---------------------------------------
def train_model(args, logger):
    lang_map = {"zu": "zulu", "xh": "xhosa", "ss": "siswati", "nr": "ndebele"}
    lang = lang_map.get(args.language, args.language)
    base_lang = "sw"

    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=base_lang, task="transcribe")

    # 8-bit config
    bits_and_bytes_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bits_and_bytes_config,
        device_map="auto"
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=base_lang, task="transcribe")
    model.config.forced_decoder_ids = forced_decoder_ids
    model.config.use_cache = False

    logger.info(f"Loading dataset for {lang}")
    dataset_path = MODEL_DIR / f"processed_arrow/{lang}"
    dataset = load_from_disk(dataset_path)
    logger.info(f"Loaded {len(dataset)} samples")

    if not isinstance(dataset, DatasetDict):
        dataset = dataset.train_test_split(test_size=0.1, seed=42)

    dataset = dataset.shuffle(seed=42)
    dataset["test"] = dataset["test"].select(range(min(100, len(dataset["test"]))))
    # ... inside train_model function ...
    
    logger.info(f"Loading dataset for {lang}")
    dataset_path = MODEL_DIR / f"processed_arrow/{lang}"
    dataset = load_from_disk(dataset_path)
    
    
    
    logger.info(f"Loaded {len(dataset)} samples")
    # ... continue with split and shuffle ...
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(MODEL_DIR / f"{args.language}_whisper_lora"),
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=1,
        learning_rate=1e-3,
        warmup_steps=50,
        weight_decay=0.01,
        fp16=True,
        logging_steps=100,
        save_steps=500,
        eval_steps=500,
        save_total_limit=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        predict_with_generate=True,
        generation_max_length=128,
        num_train_epochs=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        label_names=["labels"],
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
        compute_metrics=compute_metrics,
    )

    # Resume checkpoint
    ckpts = list(Path(training_args.output_dir).glob("checkpoint-*"))
    resume_ckpt = sorted(ckpts, key=lambda x: int(x.name.split("-")[1]))[-1] if ckpts else None
    if resume_ckpt:
        logger.info(f"⏮ Resuming from {resume_ckpt}")
        for rng in Path(resume_ckpt).glob("rng_state*.pth"):
            try: 
                rng.unlink()
            except: 
                pass

    logger.info("🚀 Starting training...")
    trainer.train(resume_from_checkpoint=str(resume_ckpt) if resume_ckpt else None)
    logger.info("✅ Training complete.")

    logger.info("💾 Saving LoRA adapter and processor...")
    model.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    # LoRA adapter only
    try:
        model.save_pretrained(training_args.output_dir, safe_serialization=True)
        model.save_pretrained_lora(training_args.output_dir)
    except:
        logger.warning("⚠️ Could not save LoRA adapter separately.")

    logger.info("📊 Running evaluation...")
    results = trainer.evaluate()
    result_path = BASE_DIR / f"results/{args.language}_whisper_lora/metrics.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"📁 Results saved to {result_path}")

# ---------------------------------------
# ENTRY POINT
# ---------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, required=True, help="Language code: zu | xh | ss | nr")
    args = parser.parse_args()

    logger = setup_logging(args.language)
    logger.info(f"🎙️ Starting LoRA ASR training for language: {args.language}")
    train_model(args, logger)
