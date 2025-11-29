import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from src.classical.templates_matching import predict_chords

AUDIO_PATH = "data/audio/let_it_be.wav"

def compute_chroma(audio_path: str):
    """
    Load audio, extract harmonic part, compute chroma.
    Returns:
        chroma: (12, N_frames)
        times: array of time positions for each frame
    """
    y, sr = librosa.load(audio_path, sr=22050)

    # Harmonic component to reduce drums / noise
    y_harm, _ = librosa.effects.hpss(y)

    hop_length = 512
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop_length)

    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)
    return chroma, times
  
def plot_chord_timeline(times, chords):
    """
    Very simple chord timeline: scatter plot of chord labels over time.
    """
    plt.figure(figsize=(12, 4))
    plt.scatter(times, chords, s=5)
    plt.title("Raw Chord Predictions (Framewise)")
    plt.xlabel("Time (s)")
    plt.ylabel("Chord")
    #plt.show()
    plt.savefig("classical_chord_timeline.png")
    
def main():
    print("Computing chroma...")
    chroma, times = compute_chroma(AUDIO_PATH)

    print("Predicting chords...")
    chord_seq = predict_chords(chroma)

    plot_chord_timeline(times, chord_seq)

    print("\nClassical baseline generated.")


if __name__ == "__main__":
    main()