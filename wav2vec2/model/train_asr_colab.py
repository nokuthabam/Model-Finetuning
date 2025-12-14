import os, json, torch, logging, re
import argparse
from pathlib import Path
from datetime import datetime
from datasets import load_dataset, concatenate_datasets, DatasetDict, Value
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
)
import torchaudio
import evaluate
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Union

# MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
MODEL_NAME = "nmoyo45/zu_wav2vec2"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
vocab_dict = {
    "|": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8, "i": 9,
    "j": 10, "k": 11, "l": 12, "m": 13, "n": 14, "o": 15, "p": 16, "q": 17, "r": 18,
    "s": 19, "t": 20, "u": 21, "v": 22, "w": 23, "x": 24, "y": 25, "z": 26,
    "[UNK]": 27, "[PAD]": 28
}

with open(MODEL_DIR / "vocab.json", "w") as f:
    json.dump(vocab_dict, f)
tokenizer = Wav2Vec2CTCTokenizer(
    vocab_file=MODEL_DIR / "vocab.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|"
)
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, 
                                             sampling_rate=16000, 
                                             padding_value=0.0, 
                                             do_normalize=True, 
                                             return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor,
                              tokenizer=tokenizer)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME,
                                       vocab_size =len(processor.tokenizer),
                                       ignore_mismatched_sizes=True,
                                       attention_dropout=0.1,
                                       hidden_dropout=0.1,
                                       feat_proj_dropout=0.0,
                                       mask_time_prob=0.05,
                                       layerdrop=0.1,
                                       ctc_loss_reduction="mean",
                                       ctc_zero_infinity=True,
                                       pad_token_id=processor.tokenizer.pad_token_id
)
wer_metric = evaluate.load("wer")
CHARS_TO_REMOVE_REGEX = r'[\,\?\.\!\-\;\:\"\“\%\‘\”\'\…\•\°\(\)\=\*\/\`\ː\’]'


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


def load_json_dataset(train_file, test_file):
    """
    Load JSON dataset for training and testing.
    """
    train_dataset = load_dataset("json", data_files=train_file)["train"]
    test_dataset = load_dataset("json", data_files=test_file)["train"]
    return DatasetDict({"train": train_dataset, "test": test_dataset})


def load_and_merge(logger, language_code):
    code_map = {
        "zu": "zulu",
        "xh": "xhosa",
        "ssw": "siswati",
        "nbl": "ndebele",
    }
    language = code_map.get(language_code)
    if not language:
        logger.error(f"Unsupported language code: {language_code}")
        raise ValueError(f"Unsupported language code: {language_code}")
    
    paths = {
        "lwazi_train": DATA_DIR / f"{language}_train.json",
        "lwazi_test": DATA_DIR / f"{language}_test.json",
        "cv_train": DATA_DIR / f"commonvoice_{language_code}_train.json",
        "nchlt_train": DATA_DIR / f"nchlt_{language}_train.json",
        "nchlt_test": DATA_DIR / f"nchlt_{language}_test.json",
    }
    
    logger.info("Loading data for {language_code}...")
    lwazi = load_json_dataset(str(paths["lwazi_train"]), str(paths["lwazi_test"]))
    sources = [lwazi["train"]]
    test_set = lwazi["test"]

    for src in ["cv_train", "nchlt_train"]:
        if paths[src].exists():
            ds = load_dataset("json", data_files={str(paths[src])})["train"]
            logger.info(f"Loaded {src} dataset for {language_code}.")

            # Convert age to string
            ds = ds.cast_column("age", Value("string"))
            sources.append(ds)
        else:
            logger.info(f"No {src} dataset found for {language_code}.")
    
    if paths["nchlt_test"].exists():
        test_set = concatenate_datasets([test_set, load_dataset("json", data_files={"train":str(paths["nchlt_test"])})["train"]])

    # Ensure age is a string (for all sources)
    for ds in sources:
        ds = ds.cast_column("age", Value("string"))
    test_set = test_set.cast_column("age", Value("string"))
    
    return DatasetDict({"train": concatenate_datasets(sources), "test": test_set})

def speech_file_to_array_fn(batch):
    """
    Load audio files and convert them to arrays.
    """
    waveform, sample_rate = torchaudio.load(batch["audio_path"])  # Load audio file 
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate,
                                                  new_freq=16000)(waveform)  # Convert to 16kHz
        sample_rate = 16000
    batch["speech"] = waveform.squeeze().numpy()  # Convert to 1D array
    # Ensure the audio is mono
    batch["sampling_rate"] = sample_rate 
    return batch


def clean_text(text):
    """
    Cleans transcript text for ASR training.
    - Lowercases
    - Removes noise tags ([n], [s], [um], etc.)
    - Strips punctuation and special characters
    - Removes stray brackets
    """

    # Lowercase and trim whitespace
    text = text.lower().strip()

    # Remove noise markers or filler tags
    text = text.replace("[n]", "").replace("[s]", "").replace("[um]", "")

    # Remove stray brackets
    text = text.replace("[", "").replace("]", "")

    # Remove unwanted punctuation and special chars
    text = re.sub(CHARS_TO_REMOVE_REGEX, "", text)

    # Normalize spaces (in case multiple spaces remain)
    text = re.sub(r"\s+", " ", text)

    return text


def prepare_dataset(batch):
    """
    Prepares the dataset by loading audio files and tokenizing text.
    """
    batch["transcript"] = clean_text(batch["transcript"])
    if isinstance(batch["speech"], list):
        input_values = processor(batch["speech"],
                                 sampling_rate=16000).input_values
        with processor.as_target_processor():
            labels = processor(batch["transcript"]).input_ids
        return {
            "input_values": input_values[0],
            "labels": labels
            }
    else:
        batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"]).input_values[0]
        batch["labels"] = processor(batch["transcript"]).input_ids
        return batch


def compute_metrics(pred):
    """
    Compute the accuracy of the model predictions.
    """
    # Get the predicted logits from the model's output.
    pred_logits = pred.predictions

    # Get the predicted token ids by taking the index with maximum probability across the last dimension of the logits tensor.
    pred_ids = np.argmax(pred_logits, axis=-1)
 
    # we replace the -100 pad with corresponding padding id
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    # Convert the predicted token ids to string
    pred_str = processor.batch_decode(pred_ids)
   

    # Convert the true label token ids to string
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    # Compute the WER metric between the predicted and true label strings 
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


@dataclass
class DataCollatorCTCWithPadding:

    processor: Wav2Vec2Processor

    # padding method used for the input sequences and defaults to True.
    padding: Union[bool, str] = True

    # the function takes in a list of features, where each feature is a dictionary containing input values and labels,
    # and returns a dictionary containing the padded input values and labels.
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        
        # extracts the input values from the features.
        input_features = [{"input_values": feature["input_values"]} for feature in features]

        # extracts the labels from the features.
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # pads the input sequences to the same length then return the result as PyTorch tensors.
        batch = self.processor.pad(input_features,padding=self.padding,return_tensors="pt", )

        
        with self.processor.as_target_processor():
          # pads the labels to the same length as the input sequences 
            labels_batch = self.processor.pad(label_features,padding=self.padding,return_tensors="pt",)

        # replaces padding in the labels with -100 so that it is ignored when calculating the loss.
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        #  sets the padded labels as the "labels" key in the batch dictionary.
        batch["labels"] = labels

        return batch
    

def train_model(language_code, logger):
    """
    Train the Wav2Vec2 model for the specified language.
    """
    # Suppress warnings from transformers library
    warnings.filterwarnings("ignore", category=UserWarning)
    
    logger.info("Processor loaded.")
    # === Device Check ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset = load_and_merge(logger, language_code)
    dataset["train"] = dataset["train"].shuffle(seed=42)
    dataset["train"] = dataset["train"].select(range(25000))  # Limit to 1000 samples for quick training
    dataset["test"] = dataset["test"].select(range(3000))  # Limit to 100 samples for quick evaluation
    dataset = dataset.map(speech_file_to_array_fn, num_proc=16)
    dataset = dataset.map(prepare_dataset)
    logger.info("Model loaded.")
    output_dir = BASE_DIR / f"model/{language_code}_wav2vec2"
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        group_by_length=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=8,
        eval_strategy="steps",
        metric_for_best_model="wer",
        greater_is_better=False,
        load_best_model_at_end=True,
        num_train_epochs=3,
        fp16=torch.cuda.is_available(),
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=5e-5,
        save_total_limit=2,
        warmup_steps=1000,
        push_to_hub=True,
        hub_model_id="nmoyo45/zu_wav2vec2",
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorCTCWithPadding(processor=processor, padding=True),
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=processor.feature_extractor,
    )
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed.")

    # === Save the model and processor ===
    logger.info("Saving model and processor...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logger.info(f"Model and processor saved to {output_dir}")


    # # === Evaluate the model ===
    # logger.info("Evaluating the model...")
    # metrics = trainer.evaluate()
    # logger.info(f"Evaluation results: {metrics}")
    # # Compute WER
    # metrics_path = BASE_DIR / f"results/{language_code}_wav2vec2/metrics.json"
    # metrics_path.parent.mkdir(parents=True, exist_ok=True)
    # with open(metrics_path, "w") as f:
    #     json.dump(metrics, f, indent=4)
    # logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Wav2Vec2 model for a specific language.")
    parser.add_argument("--language", type=str, required=True, help="Language code (e.g., zu, xh, ssw, nbl)")
    args = parser.parse_args()
    log = setup_logging(args.language)
    train_model(args.language, log)
