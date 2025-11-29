import numpy as np
from src.classical.chord_templates import CHORD_TEMPLATES, CHORD_LABELS

def normalize_chroma(chroma: np.ndarray) -> np.ndarray:
    """Normalize chroma vectors (L2 norm = 1) to compare with chord templates."""
    norm = np.linalg.norm(chroma, axis=0, keepdims=True)
    norm[norm == 0] = 1  # Prevent division by zero
    return chroma / norm
  
def framewise_similarity(chroma: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between each chord template and each chroma frame.
    - chroma: (12, N_frames), already normalized
    - CHORD_TEMPLATES: (24, 12), already normalized

    Returns:
        sims: (24, N_frames) similarity matrix
              sims[k, t] = similarity(template k, frame t)
    """
    # CHORD_TEMPLATES: (24, 12)
    # chroma: (12, N_frames)
    # result: (24, N_frames)
    return CHORD_TEMPLATES @ chroma  # matrix multiplication


def predict_chords(chroma: np.ndarray) -> list[str]:
    """
    Predict a chord label for each frame of chroma.
    - chroma: (12, N_frames)

    Returns:
        labels: list of length N_frames with chord names (e.g., 'C', 'G', 'Am', ...)
    """
    chroma_norm = normalize_chroma(chroma)
    sims = framewise_similarity(chroma_norm)  # (24, N_frames)

    # Best template index per frame
    best_idx = np.argmax(sims, axis=0)  # (N_frames,)

    # Map indices → chord labels
    labels = [CHORD_LABELS[i] for i in best_idx]
    return labels