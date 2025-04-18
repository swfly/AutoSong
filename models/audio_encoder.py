import torch
from encodec import EncodecModel
from encodec.utils import convert_audio
import torchaudio

class AudioEncoder:
    def __init__(self, device="cuda", sample_rate=24000, bandwidth=6.0):
        self.device = device
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(bandwidth)
        self.model.eval().to(self.device)
        self.sample_rate = sample_rate
        self.channels = self.model.channels

    def encode(self, wav_path: str) -> torch.Tensor:
        """
        Encode audio file into a sequence of quantized tokens.

        Returns:
            Tensor of shape (T, C) where T is number of time steps,
            and C is the number of codebooks (quantizers).
        """
        wav, sr = torchaudio.load(wav_path)
        wav = convert_audio(wav, sr, self.sample_rate, self.channels)
        wav = wav.unsqueeze(0).to(self.device)  # (1, C, T)

        with torch.no_grad():
            encoded_frames = self.model.encode(wav)

        # encoded_frames is a list of length C (num codebooks)
        # each element is (B, 1, T) → we squeeze batch and stack as (C, T)
        token_seq = torch.cat([code[0] for code in encoded_frames], dim=0)  # (C, T)
        token_seq = token_seq.permute(1, 0).contiguous()  # → (T, C)
        return token_seq  # int32 tokens

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decode token sequence back to waveform.

        Args:
            tokens: Tensor of shape (T, C) — must match Encodec's codebook count.

        Returns:
            Waveform tensor of shape (1, C, T) on CPU.
        """
        tokens = tokens.permute(1, 0).unsqueeze(0).to(self.device)  # (1, C, T)

        with torch.no_grad():
            audio = self.model.decode(tokens)

        return audio.cpu().squeeze(0)  # shape: (C, T)
