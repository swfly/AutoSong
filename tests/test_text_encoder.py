import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.text_encoder import TextEncoder

def test_text_encoder():
    encoder = TextEncoder()
    encoder.eval()
    with open("test_song.txt", encoding="utf-8") as f:
        text = f.read()
    
    token_ids = encoder.encode(text)
    print("Shape:", token_ids.shape)  # Should be [1, max_tokens]

    # Decode back to pinyin
    recovered = encoder.decode(token_ids)
    print("Decoded pinyin:", recovered)  # Should be ['wo3', 'ai4', 'ni3']

if __name__ == "__main__":
    test_text_encoder()
