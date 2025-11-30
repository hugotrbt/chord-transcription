import numpy as np
from collections import Counter

def mode_filter(sequence, window_size=9):
    """
    Apply a mode filter (majority vote) over the sequence.
    window_size must be odd.
    """
    
    half = window_size // 2
    padded = [sequence[0]] * half + sequence + [sequence[-1]] * half
    smoothed = []
    
    for i in range(len(sequence)):
        window = padded[i:i + window_size]
        most_common = Counter(window).most_common(1)[0][0]
        smoothed.append(most_common)
    
    return smoothed


def chord_segments(chord_sequence, times):
    """
    Convert framewise chord labels into chord segments with start/end times.

    Returns:
        list of dicts:
        [
            {"chord": "C", "start": 0.0, "end": 2.5},
            {"chord": "G", "start": 2.5, "end": 5.0},
            ...
        ]
    """
    
    segments = []
    current_chord = chord_sequence[0]
    start_time = times[0]
    
    for i in range(1, len(chord_sequence)):
        if chord_sequence[i] != current_chord:
            segments.append({
              "chord":current_chord, 
              "start": float(np.round(start_time, 2)),
              "end": float(np.round(times[i], 2)),
            })
            current_chord = chord_sequence[i]
            start_time = times[i]
    
    # Append the last segment
    segments.append({
        "chord": current_chord,
        "start": float(np.round(start_time, 2)),
        "end": float(np.round(times[-1], 2)),
    })
    
    return segments
    