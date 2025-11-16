import os
import json
import argparse
from pathlib import Path
import torch
import numpy
import torchaudio
from datasets import load_dataset, DatasetDict, Audio, Value, concatenate_datasets, load_from_disk
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    GenerationConfig,
    )
from datetime import datetime
import logging
from jiwer import wer
import torch.serialization
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
import evaluate

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_NAME = "openai/whisper-small"
MODEL_DIR = BASE_DIR / "model"

torch.serialization.add_safe_globals([
    (numpy._core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
    numpy.dtype,
    numpy.dtypes.Float64DType
])


def make_collator(processor):
    def whisper_data_collator(features):
        """
        Robust collator for Whisper:
        - ensures mel shape [80, time]
        - pads/crops time dim to 3000
        - dynamically pads labels
        """
        valid_feats = []

        for f in features:
            arr = f["input_features"]

            # Convert to numpy first for easy shape logic
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()
            else:
                arr = numpy.asarray(arr, dtype=numpy.float32)

            # Handle common shapes:
            # (80, T) OK
            # (T, 80) -> transpose
            # (1, 80, T) -> squeeze to (80, T)
            # anything 1-D (T,) is invalid
            if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[1] == 80:
                arr = arr[0]  # (80, T)
            elif arr.ndim == 2 and arr.shape[0] == 80:
                pass  # already (80, T)
            elif arr.ndim == 2 and arr.shape[1] == 80:
                arr = arr.T  # (T, 80) -> (80, T)
            else:
                # 1-D or unexpected 3-D, skip
                print(f"⚠️ Skipping unexpected feature shape {arr.shape}")
                continue

            if arr.ndim != 2 or arr.shape[0] != 80:
                print(f"⚠️ Skipping after reshape, got shape {arr.shape}")
                continue

            # Pad/crop time dimension to 3000
            T = arr.shape[1]
            if T < 3000:
                # pad last dimension (time)
                pad_len = 3000 - T
                arr = numpy.pad(arr, ((0, 0), (0, pad_len)), mode="constant")
            elif T > 3000:
                arr = arr[:, :3000]

            valid_feats.append(torch.tensor(arr, dtype=torch.float32))

        if not valid_feats:
            raise ValueError("All input features were invalid or empty!")

        input_features = torch.stack(valid_feats)  # [B, 80, 3000]

        # Labels: expect already-tokenized label ids in f["labels"]
        # or raw text in f["text"]. Prefer ids if present.
        if "labels" in features[0] and features[0]["labels"] is not None:
            label_features = [{"input_ids": f["labels"]} for f in features]
            labels = processor.tokenizer.pad(label_features, return_tensors="pt", padding=True).input_ids
        else:
            # fallback: tokenize text now
            texts = [f.get("text", "") for f in features]
            labels = processor.tokenizer(texts, padding=True, return_tensors="pt").input_ids

        # Mask pad tokens as -100
        if processor.tokenizer.pad_token_id is not None:
            labels[labels == processor.tokenizer.pad_token_id] = -100

        return {"input_features": input_features, "labels": labels}

    return whisper_data_collator


def setup_logging(language_code):
    """
    Set up logging configuration.
    """
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{language_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}whisper.log"
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
        "ss": "siswati",
        "nr": "ndebele"
    }

    lang_name = languages.get(language_code, language_code)
    train_file = DATA_DIR / f"{lang_name}_train.json"
    test_file = DATA_DIR / f"{lang_name}_test.json"
    cv_train = DATA_DIR / f"commonvoice_{language_code}_train.json"
    nchlt_train_file = DATA_DIR / f"nchlt_{lang_name}_train.json"
    nchlt_test_file = DATA_DIR / f"nchlt_{lang_name}_test.json"

    if not train_file.exists() or not test_file.exists():
        logger.error(f"Dataset files for {lang_name} not found.")
        raise FileNotFoundError(f"Dataset files for {lang_name} not found.")
    
    logger.info(f"Loading Lwazi dataset for {lang_name} from {train_file} and {test_file}")

    # Load Lwazi dataset
    lwazi_train = load_dataset("json", data_files={"train": str(train_file)})["train"]
    lwazi_test = load_dataset("json", data_files={"test": str(test_file)})["test"]

    # Load NCHLT dataset if available
    nchlt_train = load_dataset("json", data_files={"train": str(nchlt_train_file)})["train"] if nchlt_train_file.exists() else None
    nchlt_test = load_dataset("json", data_files={"test": str(nchlt_test_file)})["test"] if nchlt_test_file.exists() else None
    
    # Load Common Voice dataset if available
    has_cv = cv_train.exists()
    cv_ds = load_dataset("json", data_files={"train": str(cv_train)})["train"] if has_cv else None

    datasets_to_concatenate = [lwazi_train]
    if nchlt_train:
        datasets_to_concatenate.append(nchlt_train)
    if has_cv:
        datasets_to_concatenate.append(cv_ds)
    merged_ds = concatenate_datasets(datasets_to_concatenate)
    logger.info(f"Combined dataset size: {len(merged_ds)} training examples")

    # Combine test datasets if NCHLT test set is available
    test_datasets = [lwazi_test]
    if nchlt_test:
        test_datasets.append(nchlt_test)
    merged_test_ds = concatenate_datasets(test_datasets) if len(test_datasets) > 1 else test_datasets[0]
    return DatasetDict({
        "train": merged_ds,
        "test": merged_test_ds
    })


def cast_and_prepare_dataset(dataset, processor, logger):
    """
    Cast and prepare the dataset for training.
    """
    logger.info("Casting audio column and preparing dataset...")

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    def prepare(batch):
        audio_arrays = [example["array"] for example in batch["audio"]]
        sampling_rate = batch["audio"][0]["sampling_rate"] # all have same sampling rate after casting
        input_features = processor.feature_extractor(
            audio_arrays,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        ).input_features

        labels = processor.tokenizer(
            batch["transcription"],
            return_tensors="pt",
            padding="longest"
        ).input_ids

        return {
            "input_features": input_features,
            "labels": labels
        }

    dataset = dataset.map(
        prepare,
        batched=True,
        batch_size=8,  # use small batch size to conserve RAM
        num_proc=1,    # set to >1 later if needed and safe
        remove_columns=dataset["train"].column_names, 
        load_from_cache_file=False
)


    # Remove unused columns AFTER preparation
    dataset = dataset.remove_columns([col for col in dataset["train"].column_names if col not in ["input_features", "labels"]])

    return dataset


def train_model(args, logger):
    """
    Train the Whisper model for the specified language code.
    """
    logger.info(f"Loading processor and model for {args.language}...")
    language_map = {
        "zu": "zulu",
        "xh": "xhosa",
        "ss": "siswati",
        "nr": "ndebele"
    }
    language = language_map.get(args.language, args.language)
    base_lang = "sw"  # Swahili as base language for prompts
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME,
        language=base_lang,
        task="transcribe",
    )

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=base_lang,
        task="transcribe",
    )

    model.config.forced_decoder_ids = forced_decoder_ids
    model.config.use_timestamps = False
    model.config.return_timestamps = False
    model.config.use_cache = False
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.use_timestamps = False
    model.generation_config.return_timestamps = False
    model.generation_config.language = base_lang
    model.generation_config.task = "transcribe"
    model.config.suppress_tokens = []

    logger.info(f"Using {base_lang} as proxy language for {args.language}.")

    # === Device Check ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ds = load_json_dataset(args.language, logger)
    dataset = load_from_disk(MODEL_DIR / f"processed_arrow/{language}") # Load pre-processed dataset
    logger.info(f"Loaded dataset from disk with {len(dataset)} samples.")

    # if no train-test split, create one
    if not isinstance(dataset, DatasetDict):
        logger.info("No train-test split found, creating one...")
        ds = dataset.train_test_split(test_size=0.1, seed=42)
    else:
        ds = dataset
           
    ds = ds.shuffle(seed=42)
    # ds["train"] = ds["train"].select(range(25000))
    ds["test"] = ds["test"].select(range(100))  

    logger.info(f"Training dataset size: {len(ds['train'])} examples")
    logger.info(f"Testing dataset size: {len(ds['test'])} examples")

    # dataset = cast_and_prepare_dataset(dataset, processor, logger) # Avoid mapping full dataset at once
    logger.info("Dataset loaded and prepared.")

    output_dir = BASE_DIR / f"model/{args.language}_whisper"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory set to {output_dir}")
    metric = evaluate.load("wer")
    
    def compute_metrics(pred):
        
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        wer_val = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer_val}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=1,
        weight_decay=0.01,
        learning_rate=1e-5,
        lr_scheduler_type="linear",
        warmup_steps=500,
        max_steps=5000,
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        logging_steps=200,
        report_to="none",
        push_to_hub=False,
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=150,
        load_best_model_at_end=True,
        greater_is_better=False,
        metric_for_best_model="wer",
)

    whisper_data_collator_trainer = make_collator(processor)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=processor,
        data_collator=whisper_data_collator_trainer,
        compute_metrics=compute_metrics,
    )

    torch.cuda.empty_cache()

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

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=latest_checkpoint)
    logger.info("Training completed.")

    model.config.use_cache = True
    model.generation_config.use_cache = True

    # === Save the model and processor ===
    logger.info("Saving model and processor...")
    # Only save the final model to avoid large checkpoint files
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logger.info(f"Model and processor saved to {output_dir}")

    torch.cuda.empty_cache()

    # === Evaluate the model ===
    logger.info("Evaluating the model...")
    generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
    generation_config.language = "sw"
    generation_config.task = "transcribe"
    metrics = trainer.evaluate()
    logger.info(f"Evaluation metrics: {metrics}")

    # Write metrics to a JSON file
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
    print("New Session Started")
    try:
        train_model(args, logger)
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise
