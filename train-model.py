import os
import sys
import subprocess
import random
import gc
import shutil
import accelerate
import torch
import torchaudio
from torchaudio.transforms import Resample
import soundfile as sf
import librosa
import numpy as np
import pandas as pd
import transformers
from datasets import Dataset
from transformers import Seq2SeqTrainer
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments
)
import faster_whisper
import ctranslate2
from ctranslate2.converters import TransformersConverter

print(torch.__version__)
print(torch.backends.cudnn.version())
print(torch.version.cuda)
print(torch.cuda.is_available())    # should be True
print(torch.cuda.get_device_properties(0).major)  # GPU compute capability

print("cuDNN version:", torch.backends.cudnn.version())
print("cuDNN enabled:", torch.backends.cudnn.enabled)
print("cuDNN available:", torch.backends.cudnn.is_available())

print(accelerate.__version__)
print(transformers.__version__)

ROOT_DIR = "."

EXCEL_DIR = os.path.join(ROOT_DIR,"chunked_audio") # location of all xlsx
# AUDIO_DIR = "./halifax_october_gte_1.8"
AUDIO_DIR = os.path.join(ROOT_DIR,"chunked_audio")
MODEL_NAME = os.path.join(ROOT_DIR,"models/whisper-small-finetuned-30185-early-checkpoint-1385") # if going from previously trained, point it there instead
OUTPUT_DIR = os.path.join(ROOT_DIR,"models/whisper-small-30185-1385-18168") # end number = seconds of training data
NUM_EPOCHS = 5
BATCH_SIZE = 8 # 8 for small, 1 for medium
LEARNING_RATE = 1e-5
TARGET_SR = 16000
MAX_TRAIN_SAMPLES = None  # set to a number for debugging (e.g., 200)
PREPROCESS = False
WARMUP_STEPS=50

FROM_CHECKPOINT = True # False if from whisper

TUNED_INPUT = os.path.join(ROOT_DIR,"models/whisper-small-30185-1385-18168")
OUTPUT_DIR_FASTER = os.path.join(ROOT_DIR,"models/faster-whisper-small-30185-1385-18168")

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect() # Clears shared memory if using multiprocessing

print("VRAM cleared as much as possible.")

def normalize_audio(waveform, sr=16000, use_compression=False):
    # Convert to float32
    y = waveform.astype(np.float32)

    # Peak normalization
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    return y

# def reduce_noise_audio(waveform, denoise_strength, stationary=False, sr=16000):
#     y = waveform.astype(np.float32)

#     # Denoise
#     y = nr.reduce_noise(y=y, sr=sr, prop_decrease=denoise_strength, stationary=stationary)

#     return y

def preprocess_audio(waveform, sr=16000):
    """Full preprocessing chain"""
    # waveform = reduce_noise_audio(waveform, denoise_strength=0.8, stationary=False, sr=sr)
    # waveform = normalize_audio(waveform, sr=sr, use_compression=True)
    return waveform

def shuffle_records(records):
    random.shuffle(records)
    return records

def get_audio_path(file_name):
    """Resolve audio path using subfolder structure."""
    file_name = str(file_name)

    # Extract base folder name (everything before last underscore)
    if "_" in file_name:
        base_name = "_".join(file_name.split("_")[:-1])
    else:
        base_name = file_name.replace(".wav", "")

    # Construct full path
    return os.path.join(AUDIO_DIR, base_name, file_name)

# -----------------------------
# STEP 1: Load ALL Excel files
# -----------------------------
all_dfs = []

for file in os.listdir(EXCEL_DIR):
    if file.endswith(".xlsx"):
        path = os.path.join(EXCEL_DIR, file)
        print(f"Loading: {path}")
        df = pd.read_excel(path)
        all_dfs.append(df)

if not all_dfs:
    raise ValueError("No Excel files found in directory")

df = pd.concat(all_dfs, ignore_index=True)

# -----------------------------
# STEP 2: Validate columns
# -----------------------------
required_cols = {"file_name", "transcript", "Clarity"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Excel must contain columns: {required_cols}")

print(f"Total rows before filtering: {len(df)}")

# -----------------------------
# STEP 3: Filter
# -----------------------------
df = df[df["Clarity"] >= 4]
print(f"Rows after filtering: {len(df)}")

# -----------------------------
# STEP 4: Build records
# -----------------------------
records = []

for _, row in df.iterrows():
    file_name = str(row["file_name"])
    audio_path = get_audio_path(file_name)

    if os.path.exists(audio_path):
        records.append({
            "audio": audio_path,
            "sentence": str(row["transcript"])
        })
    else:
        print(f"⚠️ Missing audio file {audio_path}")

# -----------------------------
# STEP 5: Shuffle & Dataset
# -----------------------------
records = shuffle_records(records)

dataset = Dataset.from_list(records)

print(f"Final dataset size: {len(dataset)}")

# THIS VERSION HAS CONFIGURABLE PREPROCESSING
# -----------------------------
# STEP 2: Load & resample audio
# -----------------------------
def load_and_resample(record):
    waveform, sr = torchaudio.load(record["audio"])
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # mono

    waveform = waveform.squeeze().numpy()

    if PREPROCESS:
        waveform = preprocess_audio(waveform, sr=TARGET_SR)

    record["audio"] = {"array": waveform, "sampling_rate": TARGET_SR}
    return record

dataset = dataset.map(load_and_resample)

# -----------------------------
# STEP 3: Load processor & model
# -----------------------------
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
model.config.suppress_tokens = []

# -----------------------------
# STEP 4: Prepare input_features + labels
# -----------------------------
def prepare_dataset(batch):
    audio = batch["audio"]["array"]
    sr = batch["audio"]["sampling_rate"]
    features = processor.feature_extractor(audio, sampling_rate=sr)
    batch["input_features"] = features["input_features"][0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

# -----------------------------
# STEP 5: Custom collator to pad 2D features
# -----------------------------
def collate_fn(batch):
    # Pad input_features (2D)
    max_len = max(f["input_features"].shape[0] if hasattr(f["input_features"], "shape") else len(f["input_features"]) for f in batch)
    feature_dim = batch[0]["input_features"].shape[1] if hasattr(batch[0]["input_features"], "shape") else 80

    input_features = []
    for f in batch:
        feat = torch.tensor(f["input_features"], dtype=torch.float32)
        pad_len = max_len - feat.shape[0]
        if pad_len > 0:
            pad_tensor = torch.zeros((pad_len, feature_dim), dtype=torch.float32)
            feat = torch.cat([feat, pad_tensor], dim=0)
        input_features.append(feat)
    input_features = torch.stack(input_features)

    # Pad labels (1D)
    labels = [torch.tensor(f["labels"], dtype=torch.long) for f in batch]
    max_label_len = max(l.shape[0] for l in labels)
    padded_labels = []
    for l in labels:
        pad_len = max_label_len - l.shape[0]
        if pad_len > 0:
            pad_tensor = torch.full((pad_len,), -100, dtype=torch.long)
            l = torch.cat([l, pad_tensor], dim=0)
        padded_labels.append(l)
    labels = torch.stack(padded_labels)

    return {"input_features": input_features, "labels": labels}

# -----------------------------
# STEP 6: Training arguments
# -----------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    remove_unused_columns=False,
    report_to="none",
    save_total_limit=1,
    warmup_steps=WARMUP_STEPS,
    load_best_model_at_end=False
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor.feature_extractor,
    data_collator=collate_fn
)

# Optional: reset stats before training
torch.cuda.reset_peak_memory_stats()

# -----------------------------
# STEP 7: Train
# -----------------------------
trainer.train()


# Log peak GPU memory usage (in GB)
peak_memory = torch.cuda.max_memory_allocated() / 1024**3
print(f"\n🔹 Peak GPU memory usage: {peak_memory:.2f} GB")

# -----------------------------
# STEP 8: Save model
# -----------------------------
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print(f"\n✅ Fine-tuning complete! Model saved to {OUTPUT_DIR}")

# Convert model
converter = TransformersConverter(TUNED_INPUT)
converter.convert(OUTPUT_DIR_FASTER, force=True) # Removed quantization="float16"

# CRITICAL, for the fine-tuned model to work with faster-whisper, you must also copy over
# preprocessor_config.json
# tokenizer.json
# special_tokens_map.json
critical_files = [
    "preprocessor_config.json",
    "tokenizer.json",
    "special_tokens_map.json"
]

# Copy each file if it exists
for filename in critical_files:
    src = os.path.join(MODEL_NAME, filename)
    dst = os.path.join(OUTPUT_DIR_FASTER, filename)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied {filename} to {OUTPUT_DIR_FASTER}")
    else:
        print(f"Warning: {filename} not found in {MODEL_NAME}")

print("DONE")