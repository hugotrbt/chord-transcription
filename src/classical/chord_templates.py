import numpy as np

PITCH_CLASSES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
]

MAJOR_LABELS = PITCH_CLASSES.copy()
MINOR_LABELS = [pc + "m" for pc in PITCH_CLASSES]

CHORD_LABELS = MAJOR_LABELS + MINOR_LABELS  # 24 total

def create_chord_template(root_index: int, is_major: bool) -> np.ndarray:
    """
    Create a 12-D template for a major or minor chord.
    - root_index: index in PITCH_CLASSES (0–11)
    - is_major: True → major chord, False → minor chord
    """

    template = np.zeros(12)

    if is_major:
        third = (root_index + 4) % 12   # major third
    else:
        third = (root_index + 3) % 12   # minor third

    fifth = (root_index + 7) % 12       # perfect fifth

    # Put 1s at chord tones
    template[root_index] = 1
    template[third] = 1
    template[fifth] = 1

    # Normalize (L2 norm = 1) to compare with chroma using cosine similarity
    template = template / np.linalg.norm(template)

    return template

MAJOR_TEMPLATES = np.array([create_chord_template(i, True) for i in range(12)])
MINOR_TEMPLATES = np.array([create_chord_template(i, False) for i in range(12)])

# Final 24×12 matrix
# First 12 rows = major chords
# Next 12 rows = minor chords
CHORD_TEMPLATES = np.vstack([MAJOR_TEMPLATES, MINOR_TEMPLATES])

def chord_name_from_index(idx: int) -> str:
    """Return chord name corresponding to template index 0–23."""
    return CHORD_LABELS[idx]
