import whisper
import json
import argparse
import os
import torch


device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

def align_lyrics(audio_path, model_size="medium"):
    print(f"🔊 Loading Whisper ({model_size})...")
    model = whisper.load_model(model_size)

    print(f"🧠 Transcribing {audio_path}...")
    result = model.transcribe(audio_path, language="zh")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("audio", help="Path to input .wav/.mp3/.flac file")
    parser.add_argument("--out", help="Output JSON file", default="aligned_lyrics.json")
    parser.add_argument("--model", help="Whisper model size", default="medium")
    args = parser.parse_args()

    aligned = align_lyrics("dataset/song_0001/song_001.mp3", model_size=args.model)
    print(aligned)
    quit()
    
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(aligned, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved alignment to {args.out}")
