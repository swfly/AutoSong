import torch
import torchaudio
import torchaudio.transforms as T
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")

class AudioEncoder:
    def __init__(self, device="cuda", sample_rate=48000, n_mels=128, n_fft=1024, hop_length=256):
        self.device = device
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.dim = n_mels

        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            normalized=False,
            center=True,
            power=1.0
        ).to(self.device)

        # For decoding: Griffin-Lim inversion
        self.griffin_lim = T.GriffinLim(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=1.0,
            n_iter=128
        ).to(self.device)

        self.channels = 1  # Single mel channel

    def encode(self, wav_path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(wav_path)
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate).to(self.device)
            wav = resampler(wav)
        wav = wav.to(self.device)

        # If stereo, take mean to mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        mel = self.mel_transform(wav)  # (1, n_mels, T)
        mel = mel.squeeze(0)           # (n_mels, T)
        mel = mel.transpose(0,1)       # (T, n_mels)
        # Optional: apply log compression
        mel = torch.log1p(mel)
        return mel.cpu().detach()

    def decode(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (T, n_mels)
        mel = mel.to(self.device).transpose(0, 1)      # (n_mels, T)
        mel = torch.expm1(mel).unsqueeze(0)            # (1, n_mels, T)

        # --- Make the FB full-rank and use a tolerant driver ---------------
        mel_to_spec = T.InverseMelScale(
            n_stft=self.n_fft // 2 + 1,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            f_min=0.0,
            # pull f_max a bit *below* Nyquist so the last filter is non-zero
            f_max=self.sample_rate / 2 - 1.0,
            norm="slaney",                 # match your MelSpectrogram
            mel_scale="htk",               # also match           (default is "htk")
            driver="gelsy"                 # LAPACK driver that tolerates rank-loss
        ).to(self.device)
        # -------------------------------------------------------------------

        spec = mel_to_spec(mel)           # (1, n_freq, T)
        spec = spec.squeeze(0)            # (n_freq, T)

        wav  = self.griffin_lim(spec).unsqueeze(0)     # (1, T)
        print(wav.shape)
        return wav.cpu()


