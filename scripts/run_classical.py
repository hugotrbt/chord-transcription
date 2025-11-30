import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter

from src.classical.templates_matching import predict_chords
from src.classical.smoothing import mode_filter, chord_segments
from src.classical.lab_parser import parse_lab_file, harmonize_chord_labels
from src.evaluation.align import align_predictions_to_ground_truth
from src.evaluation.metrics import frame_accuracy

#AUDIO_PATH = "data/audio/let_it_be.wav"
AUDIO_PATH = "data/audio/I_ll_Follow_the_Sun.wav"

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
    plt.title("Raw Chord Predictions (Framewise) I'll Follow the Sun")
    plt.xlabel("Time (s)")
    plt.ylabel("Chord")
    #plt.show()
    plt.savefig("classical_chord_timeline_2.png")
    
def main():
    print("Computing chroma...")
    chroma, times = compute_chroma(AUDIO_PATH)

    print("Predicting chords...")
    chord_seq = predict_chords(chroma)
    
    plot_chord_timeline(times, chord_seq)
    
    smooth_chord_seq = mode_filter(chord_seq, window_size=9)
    
    segments = chord_segments(smooth_chord_seq, times)
    
    print("\nClassical baseline generated.")
    
    seg = parse_lab_file("data/annotations/let_it_be.lab")
    gt_segments = harmonize_chord_labels(seg)

    aligned_true = align_predictions_to_ground_truth(times, gt_segments)
    
    accuracy = frame_accuracy(smooth_chord_seq, aligned_true)
    print(f"Frame-wise accuracy against ground truth: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()