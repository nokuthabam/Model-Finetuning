import argparse
import json
import logging
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder
import jiwer  # Mandatory for calculating WER

# === CONFIGURATION ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
LOG_DIR = BASE_DIR / "logs/tuning"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Grid Search Range
# Alpha: Importance of the Language Model (0.0 = ignore LM, 1.0 = trust LM completely)
ALPHA_RANGE = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# Beta: Word Insertion Bonus (Higher = prefers more words / longer words)
BETA_RANGE = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

LANGUAGE_MAP = {"zu": "zulu", "xh": "xhosa", "ssw": "siswati", "nbl": "ndebele"}
LANGUAGE_MODEL_MAP = {
    "zu": MODEL_DIR / "zu_wav2vec2",
    "xh": MODEL_DIR / "xh_wav2vec2",
    "ssw": MODEL_DIR / "ssw_wav2vec2",
    "nbl": MODEL_DIR / "nbl_wav2vec2"
}
# Using specific 3-gram or 5-gram files if available
KENLM_FILE_MAP = {
    "zu": BASE_DIR / "zulu_3gram.arpa",
    "xh": BASE_DIR / "xhosa_3gram.arpa",
    "ssw": BASE_DIR / "siswati_3gram.arpa", # Or nguni_3gram.arpa
    "nbl": BASE_DIR / "ndebele_3gram.arpa"
}
CHARS_TO_REMOVE_REGEX = r'[\,\?\.\!\-\;\:\"\“\%\‘\”\'\…\•\°\(\)\=\*\/\`\ː\’]'

def clean_text(text):
    text = text.lower().strip()
    text = text.replace("[n]", "").replace("[s]", "").replace("[um]", "")
    text = text.replace("[", "").replace("]", "")
    text = re.sub(CHARS_TO_REMOVE_REGEX, "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def load_data_subset(language_code, num_samples=300):
    """Loads a small subset of validation data for quick tuning."""
    language = LANGUAGE_MAP.get(language_code)
    unseen_path = DATA_DIR / f"{language}_unseen.json"
    
    samples = []
    with open(unseen_path, 'r') as f:
        lines = f.readlines()
        
    # Use the first N samples (or shuffle if you prefer)
    for line in lines[:num_samples]:
        entry = json.loads(line)
        samples.append({
            "audio": entry["audio_path"],
            "reference": clean_text(entry["transcript"])
        })
    return samples

def get_logits(model, processor, samples, device):
    """
    Pre-calculates logits (acoustic scores) for all samples.
    We do this ONCE so we don't re-run the Neural Net for every alpha/beta.
    """
    print(f"Pre-calculating acoustic logits for {len(samples)} samples...")
    logits_list = []
    
    for sample in tqdm(samples):
        path = sample["audio"]
        if "D:\\" in path: path = path.replace("D:\\", "/mnt/d/").replace("\\", "/")
        
        # Load Audio
        try:
            waveform, sr = torchaudio.load(path)
        except:
            logits_list.append(None) # Skip bad files
            continue
            
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            
        inputs = processor(waveform.squeeze(), return_tensors="pt", sampling_rate=16000).to(device)
        with torch.no_grad():
            logits = model(inputs.input_values).logits.cpu().numpy()[0]
        
        logits_list.append(logits)
        
    return logits_list

def tune_decoder(args):
    # 1. Load Model & Processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    
    model_path = LANGUAGE_MODEL_MAP[args.language_code]
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)
    model.eval()
    
    # 2. Load Data & Pre-calculate Logits
    samples = load_data_subset(args.language_code, num_samples=args.num_samples)
    cached_logits = get_logits(model, processor, samples, device)
    
    # 3. Prepare Decoder Vocabulary
    vocab_dict = processor.tokenizer.get_vocab()
    vocab = sorted(vocab_dict.keys(), key=lambda x: vocab_dict[x])
    kenlm_path = KENLM_FILE_MAP[args.language_code]
    
    print(f"\n--- Starting Grid Search on {len(samples)} samples ---")
    print(f"LM Path: {kenlm_path}")
    print(f"Alpha Range: {ALPHA_RANGE}")
    print(f"Beta Range:  {BETA_RANGE}")
    print("-" * 60)
    print(f"{'Alpha':<10} | {'Beta':<10} | {'WER':<10}")
    print("-" * 60)
    
    best_wer = 1.0
    best_params = (0.5, 1.5)

    # 4. The Loop
    for alpha in ALPHA_RANGE:
        for beta in BETA_RANGE:
            # Re-build decoder with new params (this is fast)
            decoder = build_ctcdecoder(
                labels=vocab,
                kenlm_model_path=str(kenlm_path),
                alpha=alpha,
                beta=beta
            )
            
            hypotheses = []
            references = []
            
            # Decode using cached logits
            for i, logits in enumerate(cached_logits):
                if logits is None: continue
                
                text = decoder.decode(logits)
                hypotheses.append(text.lower())
                references.append(samples[i]["reference"])
            
            # Calculate WER
            wer = jiwer.wer(references, hypotheses)
            
            print(f"{alpha:<10} | {beta:<10} | {wer:.4f}")
            
            if wer < best_wer:
                best_wer = wer
                best_params = (alpha, beta)

    print("-" * 60)
    print(f"✅ BEST RESULT: Alpha={best_params[0]}, Beta={best_params[1]} -> WER: {best_wer:.4f}")
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language_code", type=str, required=True, choices=LANGUAGE_MAP.keys())
    parser.add_argument("--num_samples", type=int, default=300, help="Number of validation samples to test")
    args = parser.parse_args()
    
    tune_decoder(args)