def parse_lab_file(lab_path):
    """Parse a .lab file and return a list of (start, end, chord)."""
    segments = []
    with open(lab_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            start_time = float(parts[0])
            end_time = float(parts[1])
            chord_label = parts[2]
            segments.append((start_time, end_time, chord_label))
    return segments
  
def harmonize_chord_labels(segments):
    """Convert complex chord labels (A:min/b7, F:maj7, etc.) into simple major/minor."""
    harmonized_segments = []

    for start, end, label in segments:
        harmonized_label = label.split(':')[0]
        quality = label.split(':')[1] if ':' in label else 'maj'
        if 'min' in quality:
            harmonized_label += 'm'

        harmonized_segments.append(
            {"start": start, "end": end, "chord": harmonized_label}
        )
    return harmonized_segments