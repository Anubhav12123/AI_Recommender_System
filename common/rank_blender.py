from typing import Dict, Tuple
import math

# Simple weighted-sum blender; expects score dicts keyed by item_id
def blend_scores(score_dicts: Dict[str, Dict[str, float]], weights: Dict[str, float]) -> Dict[str, float]:
    # Normalize weights
    total_w = sum(max(0.0, w) for w in weights.values())
    if total_w <= 0:
        total_w = 1.0
    norm_w = {k: max(0.0, v) / total_w for k, v in weights.items()}

    combined = {}
    for source_name, scores in score_dicts.items():
        w = norm_w.get(source_name, 0.0)
        if w == 0.0: 
            continue
        for item_id, s in scores.items():
            combined[item_id] = combined.get(item_id, 0.0) + w * s
    return combined

def topk(d: Dict[str, float], k: int) -> Dict[str, float]:
    return dict(sorted(d.items(), key=lambda x: x[1], reverse=True)[:k])
