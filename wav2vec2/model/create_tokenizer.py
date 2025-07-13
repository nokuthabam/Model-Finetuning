from transformers import (Wav2Vec2CTCTokenizer, 
                          Wav2Vec2FeatureExtractor, 
                          Wav2Vec2Processor)
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def main():
    vocab_file = BASE_DIR / "data/vocab.json"

    # Create Tokenizer
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=vocab_file,
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
    processor.save_pretrained(BASE_DIR / "model/processor")


if __name__ == "__main__":
    main()
    print("Tokenizer and feature extractor created and saved successfully.")
