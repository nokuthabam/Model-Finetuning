from transformers import (Wav2Vec2CTCTokenizer, 
                          Wav2Vec2FeatureExtractor, 
                          Wav2Vec2Processor)
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model" / "processor"
CORPUS_FILE = DATA_DIR / "combined_corpus.txt"


def train_tokenizer():
    print("Training tokenizer...")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]")) # Using BPE model
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.Lowercase(),
        normalizers.StripAccents()
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=300,
        special_tokens=["[UNK]", "[PAD]", "|"]
    )
    tokenizer.train([str(CORPUS_FILE)], trainer=trainer)
    tokenizer.save(str(MODEL_DIR / "tokenizer.json"))
    print("Tokenizer trained and saved.")


def create_processor():
    print("Creating processor...")
    vocab_file = DATA_DIR / "vocab.json"

    # Create Tokenizer
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=str(vocab_file),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|"
        )

    # Create Feature Extractor
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False
        )

    # Combine into Processor
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
        )

    # Save to model/processor
    processor.save_pretrained(MODEL_DIR)
    print("Processor created and saved.")


def main():
    train_tokenizer()
    create_processor()


if __name__ == "__main__":
    main()
    print("Tokenizer and feature extractor created and saved successfully.")
