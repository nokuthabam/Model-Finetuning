import json
import torch
import torchaudio
import re
from pathlib import Path
from datasets import DatasetDict, load_dataset, concatenate_datasets, Value
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer
)
import evaluate
import argparse as arparse
import logging
from datetime import datetime
from torch.serialization import safe_globals
import numpy as np

MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

processor = None 

def setup_logging(language_code):
    """
    Set up logging configuration.
    """
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{language_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_wav2vec2.log"
    logging.basicConfig(
        level =logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class DataCollatorCTCWithPadding:
    """
    Custom data collator for Wav2Vec2 that handles padding and batching.
    """
    def __init__(self, processor, padding=True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=True,
            return_tensors="pt"
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=True,
            return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id,
            -100
        )
        batch["labels"] = labels
        return batch


def load_jsonl_dataset(train_file, test_file):
    """
    Load JSONL dataset for training and testing.
    """
    train_dataset = load_dataset("json", data_files=train_file)["train"]
    test_dataset = load_dataset("json", data_files=test_file)["train"]
    return DatasetDict({"train": train_dataset, "test": test_dataset})


def load_and_merge(logger, language_code):
    if language_code == "zu":
        language = "zulu"
    elif language_code == "xh":
        language = "xhosa"
    elif language_code == "ssw":
        language = "siswati"
    elif language_code == "nbl":
        language = "ndebele"
    else:
        logger.error(f"Unsupported language code: {language_code}")
        raise ValueError(f"Unsupported language code: {language_code}")
    # Lwazi
    lwazi_train = DATA_DIR / f"{language}_train.json"
    lwazi_test = DATA_DIR / f"{language}_test.json"

    # Common Voice
    cv_train = DATA_DIR / f"commonvoice_{language_code}_train.json"
    has_cv = cv_train.exists()

    print(f"Loading datasets for {language}...")
    logger.info(f"Loading datasets for {language_code}...")
    lwazi_ds = load_jsonl_dataset(str(lwazi_train), str(lwazi_test))

    if has_cv:
        cv_ds = load_dataset("json", data_files={"train": str(cv_train)})
        logger.info(f"Loaded Common Voice dataset for {language_code}.")
        print(f"Loaded Common Voice dataset for {language_code}.")

        # Ensure age is a string and remove non-numeric characters
        cv_ds["train"] = cv_ds["train"].cast_column(
            "age", Value("string")
        )
        lwazi_ds["train"] = lwazi_ds["train"].cast_column(
            "age", Value("string")
        )
        merged_train_ds = concatenate_datasets([lwazi_ds["train"], cv_ds["train"]])
        return DatasetDict({"train": merged_train_ds, "test": lwazi_ds["test"]})
    else:
        logger.info(f"No Common Voice dataset found for {language_code}. Using Lwazi only.")
        print(f"No Common Voice dataset found for {language_code}. Using Lwazi only.")
        return lwazi_ds


def speech_file_to_array_fn(batch):
    """
    Load audio files and convert them to arrays.
    """
    waveform, sample_rate = torchaudio.load(batch["audio_path"]) # Load audio file 
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform) # Convert to 16kHz
        sample_rate = 16000
    batch["speech"] = waveform.squeeze().numpy() # Convert to 1D array
    # Ensure the audio is mono
    batch["sampling_rate"] = sample_rate 
    return batch


def prepare_dataset(batch):
    """
    Prepares the dataset by loading audio files and tokenizing text.
    """
    batch["input_values"] = processor(batch["speech"],
                                      sampling_rate=batch["sampling_rate"]).input_values[0]
    batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids
    return batch


def compute_metrics(pred):
    """
    Compute the accuracy of the model predictions.
    """
    pred_ids = torch.from_numpy(pred.predictions).argmax(axis=-1)
    label_ids = torch.from_numpy(pred.label_ids)
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer_metric = evaluate.load("wer")
    # logger.info(f"Computed WER: {wer_metric.compute(predictions=pred_str, references=label_str)}")
    return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}


def train_model(language, language_code, logger):
    """
    Train the Wav2Vec2 model for the specified language.
    """
    global processor
    processor = Wav2Vec2Processor.from_pretrained(BASE_DIR/ "model/processor")
    # === Device Check ===
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU available. Using: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.warning("⚠️ GPU not available. Using CPU instead.")
        print("⚠️ GPU not available. Using CPU instead.")
    dataset = load_and_merge(logger, language_code)
    dataset = dataset.map(speech_file_to_array_fn)
    dataset = dataset.map(prepare_dataset)
    dataset["train"] = dataset["train"].select(range(1250))  # Limit to 1000 samples for quick training
    dataset["test"] = dataset["test"].select(range(250))  # Limit to 100 samples for quick evaluation

    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME,
                                           vocab_size =len(processor.tokenizer),
                                           ignore_mismatched_sizes=True
                                           )
    output_dir = BASE_DIR / f"model/{language_code}_wav2vec2"
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        group_by_length=True,
        per_device_train_batch_size=1,
        evaluation_strategy="epoch",
        num_train_epochs=1,
        fp16=torch.cuda.is_available(),
        # use_cpu=False,  # 
        save_strategy="epoch",
        save_steps=100,
        eval_steps=100,
        logging_steps=100,
        learning_rate=1e-4,
        save_total_limit=2,
        warmup_steps=500,
        logging_dir=BASE_DIR / "logs",
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to="none"
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

    checkpoints = [check for check in output_dir.glob("checkpoint-*") if check.is_dir()]
    latest_checkpoint = None
    if checkpoints:
        latest_checkpoint = str(sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))[-1])
        logger.info(f"Resuming from checkpoint: {latest_checkpoint}")

    # >>> Fix for PyTorch 2.6 pickle error when resuming
    with safe_globals([np.ndarray, np.float64, np.int64]):
        trainer.train(resume_from_checkpoint=latest_checkpoint)

    logger.info("Training completed.")

    # === Save the model and processor ===
    logger.info("Saving model and processor...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logger.info(f"Model and processor saved to {output_dir}")


    # === Evaluate the model ===
    logger.info("Evaluating the model...")
    metrics = trainer.evaluate()
    logger.info(f"Evaluation results: {metrics}")
    # Compute WER
    metrics_path = BASE_DIR / f"results/{language_code}_wav2vec2/metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = arparse.ArgumentParser(
        description="Train Wav2Vec2 model for ASR")
    parser.add_argument("--language", type=str, required=True, 
                        help="Language to train the model on  (e.g., 'zu', 'xh', 'ssw', 'nbl')")
    args = parser.parse_args()
    logger = setup_logging(args.language)
    logger.info(f"Starting training for {args.language} language.")
    train_model(args.language, args.language, logger)
    logger.info(f"Training completed for {args.language} language.")