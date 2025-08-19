from typing import List, Set
import math

def precision_at_k(recommended: List[str], relevant: Set[str], k: int = 10) -> float:
    rec = recommended[:k]
    hits = sum(1 for x in rec if x in relevant)
    return hits / max(1, len(rec))

def dcg_at_k(recommended: List[str], relevant: Set[str], k: int = 10) -> float:
    dcg = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        rel = 1.0 if item in relevant else 0.0
        if i == 1:
            dcg += rel
        else:
            dcg += rel / math.log2(i)
    return dcg

def idcg_at_k(k: int) -> float:
    # Ideal DCG if all top-k were relevant
    return sum(1.0 if i == 1 else 1.0 / math.log2(i) for i in range(1, k+1))

def ndcg_at_k(recommended: List[str], relevant: Set[str], k: int = 10) -> float:
    idcg = idcg_at_k(k)
    if idcg == 0: 
        return 0.0
    return dcg_at_k(recommended, relevant, k) / idcg
