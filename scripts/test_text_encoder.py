from models.text_encoder import TextEncoder

def test_text_encoder():
    encoder = TextEncoder()
    encoder.eval()

    text = "genre: lo-fi hip hop; lyrics: 'sleepy circuits hum beneath the neon code'"
    
    emb = encoder.encode(text)
    print("Embedding shape:", emb.shape)

    token_ids = encoder.tokenize(text)
    print("Token IDs:", token_ids)

    recovered = encoder.decode(token_ids)
    print("Decoded text:", recovered)

if __name__ == "__main__":
    test_text_encoder()
