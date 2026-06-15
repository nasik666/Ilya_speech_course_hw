from functools import partial

import librosa
import numpy as np
import scipy


class Sequential:
    """Chains multiple transforms together and applies them in order."""
    def __init__(self, *args):
        self.transforms = args

    def __call__(self, inp: np.ndarray):
        result = inp
        for t in self.transforms:
            result = t(result)
        return result


class Windowing:
    """
    Splits a waveform into overlapping frames.
    Default hop is half the window size (50% overlap).
    """
    def __init__(self, window_size=1024, hop_length=None):
        self.window_size = window_size
        self.hop_length = hop_length if hop_length else self.window_size // 2

    def __call__(self, waveform):
        waveform = np.asarray(waveform, dtype=float)
        win_size = self.window_size
        hop = self.hop_length

        # pad both sides so the first and last frames are centered
        pad = win_size // 2
        padded = np.pad(waveform, (pad, pad), mode='constant', constant_values=0)

        n_windows = (len(waveform) - win_size % 2) // hop + 1

        windows = np.zeros((n_windows, win_size), dtype=float)
        for i in range(n_windows):
            start = i * hop
            windows[i] = padded[start:start + win_size]

        return windows


class Hann:
    """
    Applies a Hann window to each frame to reduce spectral leakage.
    The window smoothly tapers to zero at both ends.
    """
    def __init__(self, window_size=1024):
        self.window_size = window_size
        n = np.arange(window_size)
        # cosine taper: values at edges go to zero
        self.hann_window = 0.5 * (1 - np.cos(2 * np.pi * n / window_size)).astype(np.float32)

    def __call__(self, windows):
        return windows.astype(np.float32) * self.hann_window


class DFT:
    """
    Computes the magnitude spectrum via the Discrete Fourier Transform.
    Only the first K = N//2+1 bins are kept (positive frequencies).
    """
    def __init__(self, n_freqs=None):
        self.n_freqs = n_freqs

    def __call__(self, windows):
        windows = np.asarray(windows, dtype=float)
        n_windows, N = windows.shape

        K = N // 2 + 1
        if self.n_freqs is not None:
            K = min(K, self.n_freqs)

        k = np.arange(N)
        n = np.arange(K).reshape(-1, 1)

        # DFT matrix: W[k,n] = e^(-2pi*j*k*n/N)
        W = np.exp(-2j * np.pi * n * k / N)

        spec = windows @ W.T
        return np.abs(spec)


class Square:
    """Squares each element — converts amplitude to power spectrum."""
    def __call__(self, array):
        return np.square(array)


class Mel:
    """
    Projects a linear spectrogram onto the Mel frequency scale.
    The Mel scale is perceptually motivated — closer to how humans hear pitch.
    Uses librosa's filter bank with fmin=1, fmax=8192 Hz.
    """
    def __init__(self, n_fft, n_mels=80, sample_rate=22050):
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.n_freqs = n_fft // 2 + 1

        self.mel_fb = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=1,
            fmax=8192
        ).astype(np.float64)

        # pseudo-inverse allows approximate inversion back to linear spectrum
        self.mel_fb_pinv = np.linalg.pinv(self.mel_fb)

    def __call__(self, spec):
        spec = np.asarray(spec, dtype=np.float64)
        return spec @ self.mel_fb.T

    def restore(self, mel):
        mel = np.asarray(mel, dtype=np.float64)
        return mel @ self.mel_fb_pinv.T


class GriffinLim:
    """
    Reconstructs a waveform from a magnitude spectrogram using Griffin-Lim.
    Phase is estimated iteratively (32 iterations by default).
    """
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None):
        self.griffin_lim = partial(
            librosa.griffinlim,
            n_iter=32,
            hop_length=hop_length,
            win_length=window_size,
            n_fft=window_size,
            window='hann'
        )

    def __call__(self, spec):
        return self.griffin_lim(spec.T)


class Wav2Spectrogram:
    """Full pipeline: raw waveform → magnitude spectrogram."""
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None):
        self.windowing = Windowing(window_size=window_size, hop_length=hop_length)
        self.hann = Hann(window_size=window_size)
        self.fft = DFT(n_freqs=n_freqs)
        # self.square = Square()  # uncomment to get power spectrum instead of amplitude
        self.griffin_lim = GriffinLim(window_size=window_size, hop_length=hop_length, n_freqs=n_freqs)

    def __call__(self, waveform):
        return self.fft(self.hann(self.windowing(waveform)))

    def restore(self, spec):
        return self.griffin_lim(spec)


class Wav2Mel:
    """Full pipeline: raw waveform → Mel spectrogram, with inverse."""
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None, n_mels=80, sample_rate=22050):
        self.wav_to_spec = Wav2Spectrogram(
            window_size=window_size,
            hop_length=hop_length,
            n_freqs=n_freqs
        )
        self.spec_to_mel = Mel(
            n_fft=window_size,
            n_mels=n_mels,
            sample_rate=sample_rate
        )

    def __call__(self, waveform):
        return self.spec_to_mel(self.wav_to_spec(waveform))

    def restore(self, mel):
        return self.wav_to_spec.restore(self.spec_to_mel.restore(mel))


class TimeReverse:
    """Flips the mel spectrogram along the time axis — plays the audio backwards."""
    def __call__(self, mel):
        mel = np.asarray(mel)
        return mel[::-1]


class Loudness:
    """
    Scales all mel values by a constant factor.
    factor > 1 makes it louder, factor < 1 makes it quieter.
    """
    def __init__(self, loudness_factor):
        self.factor = float(loudness_factor)

    def __call__(self, mel):
        mel = np.asarray(mel)
        return mel * self.factor


class PitchUp:
    """
    Shifts the mel spectrogram upward along the frequency axis.
    Equivalent to raising the pitch — low-frequency bins get filled with zeros.
    """
    def __init__(self, num_mels_up):
        self.shift = int(num_mels_up)

    def __call__(self, mel):
        mel = np.asarray(mel)
        T, M = mel.shape
        out = np.zeros_like(mel)

        if self.shift >= M:
            return out

        out[:, self.shift:] = mel[:, :M - self.shift]
        return out


class PitchDown:
    """
    Shifts the mel spectrogram downward along the frequency axis.
    Mirror of PitchUp — high-frequency bins get dropped, low end gets zeros.
    """
    def __init__(self, num_mels_down):
        self.shift = int(num_mels_down)

    def __call__(self, mel):
        mel = np.asarray(mel)
        T, M = mel.shape
        out = np.zeros_like(mel)

        if self.shift >= M:
            return out

        # shift content toward lower bins
        out[:, :M - self.shift] = mel[:, self.shift:]
        return out


'''
SpeedUpDown changes the number of time frames.
New length = int(speed_up_factor * original_T).
Each source frame at index `idx` maps to destination index round(idx * speed_up_factor).
'''
class SpeedUpDown:
    """
    Resamples the mel spectrogram along the time axis.
    factor > 1 → fewer frames (faster playback).
    factor < 1 → more frames (slower playback).
    """
    def __init__(self, speed_up_factor=1.0):
        self.factor = float(speed_up_factor)

    def __call__(self, mel):
        mel = np.asarray(mel)
        T, M = mel.shape
        print(T, M)

        new_T = int(self.factor * T)
        if new_T <= 0:
            new_T = 1

        out = np.zeros((new_T, M), dtype=mel.dtype)

        for idx in range(T):
            dst = round(idx * self.factor)
            if dst < new_T:
                out[dst] = mel[idx]

        return out


class FrequenciesSwap:
    """Reverses the frequency axis — bass becomes treble and vice versa."""
    def __call__(self, mel):
        mel = np.asarray(mel)
        return mel[:, ::-1]


class WeakFrequenciesRemoval:
    """
    Zeros out bins below a quantile threshold.
    Useful for suppressing background noise in the spectrogram.
    """
    def __init__(self, quantile=0.05):
        self.q = float(quantile)

    def __call__(self, mel):
        mel = np.asarray(mel)
        thresh = np.quantile(mel, self.q)
        out = mel.copy()
        out[out < thresh] = 0.0
        return out


class Cringe1:
    """
    Randomly drops mel frequency bins with probability drop_prob.
    Simulates corrupted or missing frequency channels in the audio.
    """
    def __init__(self, drop_prob=0.2):
        self.drop_prob = drop_prob

    def __call__(self, mel):
        mel = np.asarray(mel)
        T, M = mel.shape
        # random binary mask across frequency bins
        mask = (np.random.rand(M) > self.drop_prob).astype(mel.dtype)
        return mel * mask


class Cringe2:
    """
    Zeros out a random contiguous block of time frames.
    Simulates a dropout or cut in the audio — like the speaker going silent mid-sentence.
    """
    def __init__(self, max_width=10):
        self.max_width = max_width

    def __call__(self, mel):
        mel = np.asarray(mel)
        T, M = mel.shape
        out = mel.copy()

        width = np.random.randint(1, min(self.max_width, T))
        start = np.random.randint(0, T - width + 1)

        out[start:start + width] = 0
        return out
