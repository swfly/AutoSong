import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.text_encoder import TextEncoder

def test_text_encoder():
    encoder = TextEncoder()
    encoder.eval()

    # Use a real Chinese lyric
    text = "我爱你"  # "I love you"
    
    token_ids = encoder.encode(text)
    print("Token IDs:", token_ids)
    print("Shape:", token_ids.shape)  # Should be [1, max_tokens]

    # Decode back to pinyin
    recovered = encoder.decode(token_ids)
    print("Decoded pinyin:", recovered)  # Should be ['wo3', 'ai4', 'ni3']

if __name__ == "__main__":
    test_text_encoder()
