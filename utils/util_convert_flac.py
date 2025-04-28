import os
from pydub import AudioSegment

def convert_flac_to_mp3(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".mp3"):
                flac_path = os.path.join(dirpath, filename)
                mp3_path = os.path.splitext(flac_path)[0] + ".wav"

                # Skip if mp3 already exists
                if os.path.exists(mp3_path):
                    print(f"[SKIP] {mp3_path} already exists.")
                    continue
                try:
                    print(f"[CONVERT] {flac_path} -> {mp3_path}")
                    audio = AudioSegment.from_file(flac_path, format="flac")
                    audio.export(mp3_path, format="wav", bitrate="192k")
                except Exception as e:
                    print(f"[ERROR] Failed to convert {flac_path}: {e}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))  # current script location
    convert_flac_to_mp3(project_root)
