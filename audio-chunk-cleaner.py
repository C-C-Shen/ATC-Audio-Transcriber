import wave
import numpy as np
import contextlib
import os

BASE_FOLDER_INPUT = "source_audio"
BASE_FOLDER_OUTPUT = "chunked_audio"
SPECIFIC_INPUT_FOLDER = "CYOW-Tower-Ground-June-5-2026-2200Z-0330Z"
INPUT_FOLDER = os.path.join(BASE_FOLDER_INPUT, SPECIFIC_INPUT_FOLDER)
OUTPUT_FOLDER_CHUNK = os.path.join(BASE_FOLDER_OUTPUT, SPECIFIC_INPUT_FOLDER)
os.makedirs(OUTPUT_FOLDER_CHUNK, exist_ok=True)

import soundfile as sf
import librosa
import noisereduce as nr
import numpy as np

def noise_gate(y, threshold_db=-35.0, reduction_db=40.0):
    """
    Simple downward expander / noise gate.

    threshold_db: level below which signal is attenuated
    reduction_db: how much to reduce quiet parts
    """

    y = y.astype(np.float32)

    # Convert to dB
    eps = 1e-10
    magnitude = np.abs(y) + eps
    db = 20 * np.log10(magnitude)

    # Compute gain mask
    gain_db = np.where(
        db < threshold_db,
        reduction_db * (db - threshold_db) / abs(threshold_db),
        0.0
    )

    gain = 10 ** (gain_db / 20.0)

    return y * gain

def robust_rms(y):
    # ignore extreme spikes (radio static / clipping)
    low = np.percentile(y, 3)
    high = np.percentile(y, 97)
    y_clip = np.clip(y, low, high)
    return np.sqrt(np.mean(y_clip ** 2))

def normalize_audio(y, target_db=-20.0):
    y = y.astype(np.float32)

    # 1. noise gate first
    y = noise_gate(y, threshold_db=-40, reduction_db=30)

    # 2. optional: kill DC offset (helps SDR/audio streams)
    y = y - np.mean(y)

    # 3. robust RMS (prevents spikes dominating)
    rms = robust_rms(y)

    if rms < 1e-6:
        return y

    target_rms = 10 ** (target_db / 20.0)

    gain = target_rms / rms
    y = y * gain

    # 4. final soft limiter (prevents clipping artifacts)
    y = np.tanh(y)

    return y

def ensure_wav(input_path):
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".wav":
        return input_path

    if ext == ".mp3":
        wav_path = os.path.splitext(input_path)[0] + ".wav"

        print(f"Converting MP3 -> WAV: {input_path}")

        audio, sr = librosa.load(input_path, sr=None, mono=False)
        sf.write(wav_path, audio.T if audio.ndim > 1 else audio, sr)

        return wav_path

    return None

# default threshold to 300, 1000 for high noise floors
def chunk_audio_on_silence(input_wav, output_prefix, output_folder, silence_threshold=1000, silence_duration=0.7, chunk_size=4096, max_chunk_duration=29, padding=0.5):
    with contextlib.closing(wave.open(input_wav, 'rb')) as wf:
        params = wf.getparams()
        n_channels, sampwidth, framerate, n_frames = params[:4]
        audio_data = wf.readframes(n_frames)
    
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    max_chunk_samples = int(framerate * max_chunk_duration)
    padding_samples = int(framerate * padding)
    required_silence_samples = int(framerate * silence_duration)

    os.makedirs(output_folder, exist_ok=True)

    chunks = []
    start = 0
    min_chunk_samples = int(framerate * 2)
    while start < len(audio_array):
        end = min(start + max_chunk_samples, len(audio_array))
        best_cut = end

        # chunk on last
        # for i in range(start + chunk_size, end, chunk_size):
        #     chunk = audio_array[i:i+chunk_size]
        #     if np.max(np.abs(chunk)) <= silence_threshold:
        #         best_cut = i

        silent_samples = 0

        # chunk on first
        for i in range(start + min_chunk_samples, end, chunk_size):
            chunk = audio_array[i:i+chunk_size]

            if np.max(np.abs(chunk)) <= silence_threshold:
                silent_samples += chunk_size
            else:
                silent_samples = 0

            if silent_samples >= required_silence_samples:
                best_cut = i
                break

        if ((best_cut - start) / framerate) < 2:
            print(f"Skipping {len(chunks) + 1} in {input_wav}")
            start = best_cut
            continue

        chunk_data = audio_array[start:best_cut]

        # create silent padding
        pad = np.zeros(padding_samples, dtype=np.int16)

        # add silence before and after
        chunk_data = np.concatenate([pad, chunk_data, pad])

        # Normalize audio
        chunk_float = normalize_audio(chunk_data)
        chunk_data = (chunk_float * 32767).astype(np.int16)

        chunk_filename = os.path.join(
            output_folder,
            f"{os.path.splitext(os.path.basename(output_prefix))[0]}_{len(chunks)+1}.wav"
        )

        with wave.open(chunk_filename, 'wb') as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(framerate)
            wf.writeframes(chunk_data.tobytes())

        print(f"Saved chunk {len(chunks)+1} to {chunk_filename}")
        chunks.append((start, best_cut))
        start = best_cut

# default threshold to 300, 1000 for high noise floors
def remove_silence_with_buffer(input_wav, output_wav, silence_threshold=1000, chunk_size=4096, buffer_duration=1):
    with contextlib.closing(wave.open(input_wav, 'rb')) as wf:
        params = wf.getparams()
        n_channels, sampwidth, framerate, n_frames = params[:4]
        audio_data = wf.readframes(n_frames)

    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    buffer_size = int(framerate * buffer_duration)

    non_silent_ranges = []
    for i in range(0, len(audio_array), chunk_size):
        chunk = audio_array[i:i+chunk_size]
        if np.max(np.abs(chunk)) > silence_threshold:
            start = max(0, i - buffer_size)
            end = min(len(audio_array), i + chunk_size + buffer_size)
            non_silent_ranges.append((start, end))

    merged_ranges = []
    for start, end in sorted(non_silent_ranges):
        if not merged_ranges or start > merged_ranges[-1][1]:
            merged_ranges.append([start, end])
        else:
            merged_ranges[-1][1] = max(merged_ranges[-1][1], end)

    if not merged_ranges:
        print(f"No non-silent audio detected in {input_wav}")
        return

    processed_audio = np.concatenate([audio_array[start:end] for start, end in merged_ranges])

    with wave.open(output_wav, 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(processed_audio.tobytes())

    print(f"Processed audio saved to {output_wav}")

def get_audio_duration(filepath):
    with wave.open(filepath, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)

for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(('.wav', '.mp3')):
        input_path = os.path.join(INPUT_FOLDER, filename)

        # Convert mp3 -> wav if needed
        input_path = ensure_wav(input_path)

        if input_path is None:
            continue

        output_cleaned = os.path.join(OUTPUT_FOLDER_CHUNK, f"{filename}")
        output_location = os.path.join(OUTPUT_FOLDER_CHUNK, f"{filename}").replace(".wav", "")

        print(f"\nProcessing {filename}...")
        remove_silence_with_buffer(input_path, output_cleaned)

        duration = get_audio_duration(output_cleaned)
        if duration < 30:
            print(f"Skipping chunking for {filename} (duration: {duration:.2f}s)")
            new_path = os.path.join(OUTPUT_FOLDER_CHUNK, os.path.basename(output_location))
            os.rename(output_location, new_path)
            continue

        chunk_audio_on_silence(output_cleaned, output_location, OUTPUT_FOLDER_CHUNK)