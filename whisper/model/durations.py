import torchaudio
from datasets import load_from_disk
from tqdm import tqdm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "model" / "processed_arrow" / "zulu"


print("Loading dataset from disk...")
dataset = load_from_disk(DATASET_PATH)
print(f"Loaded dataset from disk with {len(dataset)} samples.")

def compute_duration(batch):
    """
    Adds 'duration' (in seconds) to each example using torchaudio.
    Works even if the dataset is an Arrow shard.
    """
    audio_info = batch.get("audio", None)
    if not audio_info:
        batch["duration"] = None
        return batch

    # Handle both dict-style and string paths
    audio_path = audio_info.get("path") if isinstance(audio_info, dict) else None
    if not audio_path:
        batch["duration"] = None
        return batch

    try:
        info = torchaudio.info(audio_path)
        batch["duration"] = info.num_frames / info.sample_rate
    except Exception as e:
        # If torchaudio fails (corrupted file, etc.)
        batch["duration"] = None
    return batch


# ---------------------------------------------------------------
# 🔹 Step 2: Compute duration for all samples (parallelized)
# ---------------------------------------------------------------
print("Computing audio durations using torchaudio (this may take a few minutes)...")
dataset = dataset.map(compute_duration, num_proc=4, desc="Computing durations")

# Check how many have valid durations
valid_count = sum(1 for d in dataset["duration"] if d is not None)
print(f"Valid durations: {valid_count}/{len(dataset)} examples")


# ---------------------------------------------------------------
# 🔹 Step 3: Split dataset and filter by duration range
# ---------------------------------------------------------------
print("Creating train-test split...")
dataset = dataset.train_test_split(test_size=0.1, seed=42)

print("Filtering dataset to include only audio between 1 and 30 seconds...")
def filter_duration(batch):
    dur = batch.get("duration", None)
    return dur is not None and 1.0 <= dur <= 30.0

dataset["train"] = dataset["train"].filter(filter_duration, num_proc=4, desc="Filtering train set")
dataset["test"] = dataset["test"].filter(filter_duration, num_proc=4, desc="Filtering test set")

print(f"✅ Post-filtering: train={len(dataset['train'])}, test={len(dataset['test'])}")

# ---------------------------------------------------------------
# 🔹 Step 4 (Optional): Log duration stats
# ---------------------------------------------------------------
durations = [d for d in dataset["train"]["duration"] if d is not None]
if durations:
    print(f"After filtering — Shortest: {min(durations):.2f}s | "
                f"Longest: {max(durations):.2f}s | "
                f"Mean: {sum(durations)/len(durations):.2f}s | "
                f"Median: {sorted(durations)[len(durations)//2]:.2f}s")
else:
    print("⚠️ No durations available after filtering.")