import numpy as np
import torch
from librosa.util import frame
from scipy.signal.windows import hamming
from torchaudio.functional import melscale_fbanks


class LogMelFilterBankSpectrum(object):
    """
    Transform a signal into it's log-mel filter bank spectrum.
    """

    def __init__(self, frame_length, hop_length):
        self.hop_length = hop_length
        self.frame_length = frame_length

    def __call__(self, sample):

        # Convert from torch to numpy
        sample = np.squeeze(sample.cpu().numpy())

        # Add pre-emphasis ?

        # Frame
        frames = frame(
            sample, frame_length=self.frame_length, hop_length=self.hop_length
        )

        # Window
        frames = torch.tensor(np.multiply(hamming(frames.shape[1]), frames))

        # Get magnitude spectrum via FFT
        n_fft = 1024
        spectrums = torch.abs(torch.fft.rfft(frames, n=n_fft, axis=1)).float()

        # Do we need to convert from magnitude spectrum to power spectrum ?
        # pow_frames = ((1.0 / n_fft) * ((spectrums) ** 2))

        # Create mel-scale filter banks
        n_mels = 64
        sample_rate = 16000
        mel_filters = melscale_fbanks(
            n_freqs=int(n_fft // 2 + 1),
            f_min=0.0,
            f_max=sample_rate / 2.0,
            n_mels=n_mels,
            sample_rate=sample_rate,
            # norm="slaney",
        )

        # Convert the spectrum's frequency axis to mel-scale
        mel_spectrums = torch.mm(mel_filters.T, spectrums.T)

        # Convert the spectrum's amplitude axis to log-scale
        log_mel_spectrums = torch.log(mel_spectrums).T

        import matplotlib.pyplot as plt

        plt.plot(log_mel_spectrums[0])
        plt.savefig("log_mel_spectrums.png", dpi=300)

        print(log_mel_spectrums.shape)

        return log_mel_spectrums
