import os
import torchaudio
import torch
import soundfile as sf
from openunmix import predict
from tqdm import tqdm  # <--- for progress bar

# Choose device
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

device = torch.device("cpu")

# Root dataset folder
DATASET_ROOT = "dataset"
AUDIO_EXTS = {".mp3", ".wav", ".flac"}

# Gather all song directories that contain an audio file
song_dirs = []
for song_dir in sorted(os.listdir(DATASET_ROOT)):
    path = os.path.join(DATASET_ROOT, song_dir)
    if not os.path.isdir(path):
        continue
    for fname in os.listdir(path):
        if "vocals" in fname or "accomp" in fname:
            continue
        if os.path.splitext(fname)[1].lower() in AUDIO_EXTS:
            song_dirs.append((path, fname))
            print(path,fname)
            break

# Process each song with tqdm progress bar
for path, fname in tqdm(song_dirs, desc="Separating stems", unit="song"):
    audio_file = os.path.join(path, fname)

    try:
        # 1. Load audio
        waveform, sr = torchaudio.load(audio_file)
        waveform = waveform.to(torch.float32)

        # 2. Separate
        estimates = predict.separate(
            audio=waveform,
            rate=sr,
            model_str_or_path="umxl",
            targets=["vocals", "other"],
            residual=True,
            device=device
        )

        # 3. Aggregate + convert
        vocals = estimates["vocals"].squeeze().cpu().permute(1, 0).numpy()
        accomp = estimates["other"].squeeze().cpu().permute(1, 0).numpy()

        # 4. Save
        sf.write(os.path.join(path, "vocals.wav"), vocals, sr)
        sf.write(os.path.join(path, "accompaniment.wav"), accomp, sr)

    except Exception as e:
        print(f"\n❌ Error processing {path}: {e}")
