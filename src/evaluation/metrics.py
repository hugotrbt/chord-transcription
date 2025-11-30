def frame_accuracy(pred, true):
    """Return fraction of frames where pred[i] == true[i]."""
    
    correct = sum(p == t for p, t in zip(pred, true))
    return correct / len(true) if true else 0.0
  
