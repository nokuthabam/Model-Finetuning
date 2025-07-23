import os
import json
import argparse
from pathlib import Path
import torch
import torchaudio
from datasets import load_dataset, DatasetDict, Audio, Value, concatenate_datasets
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    GenerationConfig
    )
from datetime import datetime
import logging
from jiwer import wer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_NAME = "openai/whisper-small"


def make_collator(processor):
    def whisper_data_collator(features):
        """
        Custom data collator for Whisper model.
        """
        input_features = torch.stack([
            torch.tensor(feature["input_features"]) if isinstance(feature["input_features"], list)
            else feature["input_features"]
            for feature in features
            ])
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels = processor.tokenizer.pad(
            label_features, return_tensors="pt", padding=True
        ).input_ids
        labels[labels == processor.tokenizer.pad_token_id] = -100  # Ignore padding in loss
        return {
            "input_features": input_features,
            "labels": labels
        }
    return whisper_data_collator


def setup_logging(language_code):
    """
    Set up logging configuration.
    """
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{language_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_wav2vec2.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_json_dataset(language_code, logger):
    """
    Load JSON dataset for the specified language code.
    """
    languages = {
        "zu": "zulu",
        "xh": "xhosa",
        "ss": "swati",
        "nr": "ndebele"
    }

    lang_name = languages.get(language_code, language_code)
    train_file = DATA_DIR / f"{lang_name}_train.json"
    test_file = DATA_DIR / f"{lang_name}_test.json"
    cv_train = DATA_DIR / f"commonvoice_{language_code}_train.json"

    if not train_file.exists() or not test_file.exists():
        logger.error(f"Dataset files for {lang_name} not found.")
        raise FileNotFoundError(f"Dataset files for {lang_name} not found.")
    
    logger.info(f"Loading Lwazi dataset for {lang_name} from {train_file} and {test_file}")
    lwazi_train = load_dataset("json", data_files={"train": str(train_file)})["train"]
    lwazi_test = load_dataset("json", data_files={"test": str(test_file)})["test"]

    has_cv = cv_train.exists()
    if has_cv:
        logger.info(f"Loading Common Voice dataset for {lang_name} from {cv_train}")
        cv_ds = load_dataset("json", data_files={"train": str(cv_train)})["train"]
        merged_ds = concatenate_datasets([lwazi_train, cv_ds])
        logger.info(f"Combined dataset size: {len(merged_ds)} training examples")
    else:
        merged_ds = lwazi_train
        logger.info(f"Using dataset size: {len(merged_ds)} training examples")
    return DatasetDict({
        "train": merged_ds,
        "test": lwazi_test
    })


def cast_and_prepare_dataset(dataset, processor, logger):
    """
    Cast and prepare the dataset for training.
    """
    logger.info("Casting audio column and preparing dataset...")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    def prepare(example):
        audio = example["audio"]
        input_features = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt"
        ).input_features[0]
        labels = processor.tokenizer(
            example["transcription"],
            return_tensors="pt",
            padding="longest",
        ).input_ids.squeeze(0)

        return {
            "input_features": input_features,
            "labels": labels
        }

    dataset = dataset.map(prepare, remove_columns=["audio"])
    return dataset


def train_model(args, logger):
    """
    Train the Whisper model for the specified language code.
    """
    logger.info(f"Loading processor and model for {args.language}...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="sw", task="transcribe")
    model.config.suppress_tokens = []

    logger.info("Loading dataset...")
    ds = load_json_dataset(args.language, logger)
    ds = cast_and_prepare_dataset(ds, processor, logger)
    ds["train"] = ds["train"].select(range(1000))  # Limit to 1000 samples for quick training
    ds["test"] = ds["test"].select(range(100))  # Limit to 100 samples for quick evaluation

    logger.info("Dataset loaded and prepared.")

    output_dir = BASE_DIR / f"model/{args.language}_whisper"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory set to {output_dir}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=1,
        learning_rate=1e-4,
        logging_dir=str(BASE_DIR / "logs"),
        predict_with_generate=True,
        generation_max_length=225,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    whisper_data_collator_trainer = make_collator(processor)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=processor.feature_extractor,
        data_collator=whisper_data_collator_trainer
    )
    latest_checkpoint = None
    checkpoints = [check for check in output_dir.glob("checkpoint-*") if check.is_dir()]

    if checkpoints:
        latest_checkpoint = str(sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))[-1])
        logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
    
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=latest_checkpoint)
    logger.info("Training completed.")

    # === Save the model and processor ===
    logger.info("Saving model and processor...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logger.info(f"Model and processor saved to {output_dir}")

    # === Evaluate the model ===
    logger.info("Evaluating the model...")
    generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
    generation_config.language = "sw"
    generation_config.task = "transcribe"
    metrics = trainer.evaluate(generation_config=generation_config)
    # Compute WER
    
    pred = trainer.predict(ds["test"], generation_config=generation_config)
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer_metric = wer(pred_str, label_str)

    # Store Metrics
    metrics["wer"] = wer_metric
    metric_path = BASE_DIR / f"results/{args.language}_whisper/metrics.json"
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metric_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {metric_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Whisper model for ASR")
    parser.add_argument("--language", type=str, required=True, help="Language code for training")
    args = parser.parse_args()

    logger = setup_logging(args.language)
    logger.info(f"Starting training for language: {args.language}")

    try:
        train_model(args, logger)
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise
