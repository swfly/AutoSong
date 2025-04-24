import torch
from encodec import EncodecModel
from encodec.utils import convert_audio
import torchaudio
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")
class AudioEncoder:
    def __init__(self, device="cuda", sample_rate=48000, bandwidth=1.5):
        if sample_rate == 48000:
            self.model = EncodecModel.encodec_model_48khz()
        elif sample_rate == 24000:
            self.model = EncodecModel.encodec_model_24khz()
        else:
            raise Exception("unsupported sampling rate")
        self.device = device
        self.model.set_target_bandwidth(bandwidth)
        self.model.eval().to(self.device)
        self.sample_rate = sample_rate
        self.channels = self.model.channels  # Number of channels for the model (quantizers)
        
        # Access the codebook and determine the vocabulary size
        try:
            # Iterate over each vector quantization layer in the quantizer
            quantizer = self.model.quantizer.vq
            first_codebook = quantizer.layers[0]._codebook  # Assuming all codebooks have the same size
            
            # The vocabulary size is the first dimension of the codebook
            self.vocab_size = first_codebook.codebook_size  # This gives the number of quantization vectors
        except Exception as e:
            self.vocab_size = None

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
        codebooks = encoded_frames[0][0]  # Shape: (1, C, T)
        codebooks = codebooks.squeeze(0).permute(1, 0)  # → (T, C)
        return codebooks.int().cpu().detach()  # int32 tokens

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.to(self.device).int()                      # (T, K)
        codes  = tokens.transpose(0, 1).unsqueeze(0)               # (1, K, T)

        encoded_frames = [(codes, getattr(self, "_last_scale", None))]
        with torch.no_grad():
            audio = self.model.decode(encoded_frames)              # (1, C, T)

        return audio.squeeze(0).cpu()
