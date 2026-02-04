import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load OGG audio files
audio1, sr1 = librosa.load("audio files/reference.wav.ogg", sr=None)
audio2, sr2 = librosa.load("audio files/pattern.wav.ogg", sr=None)

# Convert to mono if stereo
if audio1.ndim > 1:
    audio1 = np.mean(audio1, axis=1)
if audio2.ndim > 1:
    audio2 = np.mean(audio2, axis=1)

# FFT
fft1 = np.abs(np.fft.fft(audio1))
fft2 = np.abs(np.fft.fft(audio2))

# Match lengths
min_len = min(len(fft1), len(fft2))
fft1 = fft1[:min_len]
fft2 = fft2[:min_len]

# Similarity (cosine similarity)
similarity = np.dot(fft1, fft2) / (np.linalg.norm(fft1) * np.linalg.norm(fft2))

print("Similarity Score:", similarity)

# Plot
plt.figure()
plt.plot(fft1, label="Reference")
plt.plot(fft2, label="Pattern")
plt.legend()
plt.title("Frequency Comparison")
plt.show()
