import torch
import torchaudio
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.audio_encoder import AudioEncoder

import matplotlib.pyplot as plt

def test_audio_encoder(wav_path: str, output_path: str = "reconstructed.wav"):
    assert os.path.exists(wav_path), f"File not found: {wav_path}"

    print(f"🔍 Loading and encoding: {wav_path}")
    encoder = AudioEncoder(device="cuda" if torch.cuda.is_available() else "cpu")

    tokens = encoder.encode(wav_path)
    print(f"✅ Encoded tokens shape: {tokens.shape}  (Time steps x Mel specturms)")
    # print(tokens)
    # bins = tokens[100].detach().numpy()
    # plt.bar(range(len(bins)), bins)
    # plt.show()
    # quit()
    print("🔁 Decoding back to waveform...")
    audio = encoder.decode(tokens[:4096])

    print(f"💾 Saving reconstructed audio to: {output_path}")
    torchaudio.save(output_path, audio, encoder.sample_rate)

    print("🎧 Done. You can now listen to the output.")

if __name__ == "__main__":
    test_audio_encoder("dataset/song_001/song_001.mp3", "output.wav")
