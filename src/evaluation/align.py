def align_predictions_to_ground_truth(times, gt_segments):
    """
    For each frame time, find which ground-truth segment it falls into.
    Return a list of true chords aligned with the predicted frames.
    """
    aligned_true = []

    for t in times:
        matched = False
        for seg in gt_segments:
            if seg["start"] <= t < seg["end"]:
                aligned_true.append(seg["chord"])
                matched = True
                break
        if not matched:
            aligned_true.append("N")  # No chord / silence

    return aligned_true