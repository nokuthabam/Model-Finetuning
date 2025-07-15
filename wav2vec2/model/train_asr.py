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

MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

processor = None 


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


def load_and_merge(language_code):
    if language_code == "zu":
        language = "zulu"
    elif language_code == "xh":
        language = "xhosa"
    elif language_code == "ssw":
        language = "siswati"
    elif language_code == "nbl":
        language = "ndebele"
    # Lwazi
    lwazi_train = DATA_DIR / f"{language}_train.json"
    lwazi_test = DATA_DIR / f"{language}_test.json"

    # Common Voice
    cv_train = DATA_DIR / f"commonvoice_{language_code}_train.json"
    has_cv = cv_train.exists()

    print(f"Loading datasets for {language}...")
    lwazi_ds = load_jsonl_dataset(str(lwazi_train), str(lwazi_test))

    if has_cv:
        cv_ds = load_dataset("json", data_files={"train": str(cv_train)})
        print(f"Loaded Common Voice dataset for {language_code}.")

        #Ensure age is a string and remove non-numeric characters
        cv_ds["train"] = cv_ds["train"].cast_column(
            "age", Value("string")
        )
        lwazi_ds["train"] = lwazi_ds["train"].cast_column(
            "age", Value("string")
        )
        merged_train_ds = concatenate_datasets([lwazi_ds["train"], cv_ds["train"]])
        return DatasetDict({"train": merged_train_ds, "test": lwazi_ds["test"]})
    else:
        print(f"No Common Voice dataset found for {language_code}. Using Lwazi only.")
        return lwazi_ds


def speech_file_to_array_fn(batch):
    """
    Load audio files and convert them to arrays.
    """
    waveform, sample_rate = torchaudio.load(batch["audio_path"])
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        sample_rate = 16000
    batch["speech"] = waveform.squeeze().numpy()
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
    pred_ids = torch.argmax(pred.predictions, dim=-1)
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer_metric = evaluate.load("wer")
    return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}


def train_model(language, language_code):
    """
    Train the Wav2Vec2 model for the specified language.
    """
    global processor
    processor = Wav2Vec2Processor.from_pretrained(BASE_DIR/ "model/processor")

    dataset = load_and_merge(language_code)
    dataset = dataset.map(speech_file_to_array_fn)
    dataset = dataset.map(prepare_dataset)

    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME,
                                           vocab_size =len(processor.tokenizer),
                                           ignore_mismatched_sizes=True
                                           )

    training_args = TrainingArguments(
        output_dir=BASE_DIR / f"model/{language_code}_wav2vec2",
        group_by_length=True,
        per_device_train_batch_size=8,
        evaluation_strategy="epoch",
        num_train_epochs=10,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        save_steps=400,
        eval_steps=400,
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

    trainer.train(resume_from_checkpoint=True if (BASE_DIR / f"model/{language_code}_wav2vec2/checkpoint").exists() else None)

if __name__ == "__main__":
    parser = arparse.ArgumentParser(
        description="Train Wav2Vec2 model for ASR")
    parser.add_argument("--language", type=str, required=True, 
                        help="Language to train the model on  (e.g., 'zu', 'xh', 'ssw', 'nbl')")
    args = parser.parse_args()
    train_model(args.language, args.language)
    print(f"Training completed for {args.language} language.")