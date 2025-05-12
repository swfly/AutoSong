import whisper
import json
import argparse
import os
import torch
import torchaudio
import re


device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

def detect_silence_duration(audio_tensor, sample_rate, silence_thresh=1.0, min_silence_len=1000):
    # Convert silence_thresh to amplitude (this is a rough threshold in dB)

    # Calculate the energy of the audio signal
    audio_energy = audio_tensor.abs().mean(dim=1)
    silence_thresh = audio_energy.max() * 0.35
    # Identify where the audio is below the silence threshold
    silent_frames = audio_energy < silence_thresh
    # Find the first non-silent frame
    first_non_silent_frame = (silent_frames == False).nonzero().min() if (silent_frames == False).any() else 0
    # Convert frames to time in seconds
    silence_duration = first_non_silent_frame / sample_rate
    return silence_duration

def open_text_file(audio_path):
    # Get the directory containing the audio file
    audio_dir = os.path.dirname(audio_path)
    
    # List all files in the directory
    files_in_directory = os.listdir(audio_dir)
    
    # Find the .txt file in the directory
    txt_files = [file for file in files_in_directory if file.endswith('.txt')]
    
    if len(txt_files) == 1:
        # Open the only .txt file found
        txt_file_path = os.path.join(audio_dir, txt_files[0])
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            # Read and return the content of the .txt file
            return f.read()
    else:
        # Return an error if no .txt file or more than one .txt file is found
        raise FileNotFoundError(f"Expected exactly one .txt file in the directory, but found {len(txt_files)}.")

def align_lyrics(audio_path, model):
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.permute(1,0) #[T,2]
    silence = int(detect_silence_duration(waveform, sample_rate, silence_thresh=0.2).numpy())
    print('silence time',silence)
    print(f"🧠 Transcribing {audio_path}...")
    prompt = "简体中文"
    result = model.transcribe(audio_path, clip_timestamps = [float(silence)], language="zh", no_speech_threshold = 0.8, verbose=True, initial_prompt=prompt)

    segments = result["segments"]
    aligned = [
        {
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "text": seg["text"].strip()
        }
        for seg in segments
    ]
    return aligned

def get_song_list(dataset_dir="dataset", max_songs=None):
    def song_number(s):
        match = re.search(r"song_(\d+)", s)
        return int(match.group(1)) if match else float("inf")

    all_dirs = [d for d in os.listdir(dataset_dir)
                if os.path.isdir(os.path.join(dataset_dir, d)) and d.startswith("song_")]

    sorted_dirs = sorted(all_dirs, key=song_number)
    if max_songs is not None:
        sorted_dirs = sorted_dirs[:max_songs]

    return sorted_dirs

def get_song_path(path: str):
    
    audio_file = next(f for f in os.listdir(path) if f.endswith((".wav", ".mp3", ".flac")) and f.startswith(("song_")))
    audio_path = os.path.join(path, audio_file)
    return audio_path

if __name__ == "__main__":
    model_size = "medium"
    print(f"🔊 Loading Whisper ({model_size})...")
    model = whisper.load_model(model_size).to(device)
    DATASET_DIR = "dataset"
    songs = get_song_list(DATASET_DIR)
    for song in songs:
        audio = get_song_path(os.path.join(DATASET_DIR, song))
    
        aligned = align_lyrics(audio, model)
        audio_dir = os.path.dirname(audio)
        out_path = os.path.join(audio_dir, "aligned_lyrics.json")
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(aligned, f, ensure_ascii=False, indent=2)
    
        print(f"✅ Saved alignment to {out_path}")
