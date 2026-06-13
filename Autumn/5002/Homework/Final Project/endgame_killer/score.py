''' Module for all score-related helper functions.'''
import numpy as np
import detector
from collections import OrderedDict

# Simple bounded caches to avoid recomputing expensive detector/score
# work for identical board+color inputs. Size chosen conservatively.
_CACHE_SIZE = 128
_raw_cache: "OrderedDict" = OrderedDict()
_kernel_cache: "OrderedDict" = OrderedDict()

def reset_score_cache():
    """Clear internal score caches (useful for tests/benchmarks)."""
    _raw_cache.clear()
    _kernel_cache.clear()


def raw_scores(board, color):
    ''' Calculate the raw score for the player of `color` according to the given board.
    The raw score is depending on the number of threats of different types that are 
    affecting the position (u, v).
    
    Two weight profiles: aggressive for Black (color==1), more defensive for White (color==-1)
    Weights formatted as: 'W_<threat_type>[_BLOCK]' where:
    - <threat_type> is one of: The name of the detected threat type
    - _Block suffix indicates the threat is given by the opponent and thus blocking it is considered

    returns:
        scores: dict mapping (u,v) positions to their computed raw score
    '''

    black_weights = {
        'W_type4': 100_000,  # Immediate win
        'W_type4_Block': 100_000,
        
        'W_Lethal3': 10_000,
        'W_Lethal3_Block': 8_000,
        'W_NonLethal3': 500,
        'W_NonLethal3_Block': 200,
        'W_Type2': 50,
        'W_Type2_Block': 30,
        'W_WeakType2': 10,
        'W_WeakType2_Block': 5,
        'W_Surround': 1
    }

    white_weights = {
        'W_type4': 100_000,  # Immediate win
        'W_type4_Block': 100_000,
        
        'W_Lethal3': 10_000,
        'W_Lethal3_Block': 9_000,
        'W_NonLethal3': 500,
        'W_NonLethal3_Block': 400,
        'W_Type2': 50,
        'W_Type2_Block': 40,
        'W_WeakType2': 10,
        'W_WeakType2_Block': 8,
        'W_Surround': 1
    }

    weights = black_weights if color == 1 else white_weights

    own = color
    opp = -color

    # try module-level cache first
    try:
        key = (board.tobytes(), int(color))
    except Exception:
        key = None
    if key is not None:
        cached = _raw_cache.get(key)
        if cached is not None:
            return dict(cached)

    n = board.shape[0]
    scores = {}

    # Sets of moves for different threat types

    # Set of lethal type4 moves
    type4_moves_own = detector.lethal_type4_moves(board, own)
    type4_moves_opp = detector.lethal_type4_moves(board, opp)

    # Set of lethal type3 moves
    type3_moves_own = detector.lethal_type3_moves(board, own)
    type3_moves_opp = detector.lethal_type3_moves(board, opp)

    # Set of non-lethal type3 moves
    nonlethal3_moves_own = detector.nonlethal_type3_moves(board, own)
    nonlethal3_moves_opp = detector.nonlethal_type3_moves(board, opp)

    # Set of type2 moves
    type2_moves_own = detector.type2_moves(board, own)
    type2_moves_opp = detector.type2_moves(board, opp)

    # Set of weak type2 moves
    weaktype2_moves_own = detector.weak_type2_moves(board, own)
    weaktype2_moves_opp = detector.weak_type2_moves(board, opp)
    
    # iterate legal cells
    for u in range(n):
        for v in range(n - u):
            # skip non-playable or occupied
            if not detector._is_valid(board, u, v):
                continue
            
            score = 0 # initialize score for this position

            if (u, v) in type4_moves_own:
                score += weights['W_type4']
            if (u, v) in type4_moves_opp:
                score += weights['W_type4_Block']



            if (u, v) in type3_moves_own:
                score += weights['W_Lethal3']
            if (u, v) in type3_moves_opp:
                score += weights['W_Lethal3_Block']

            if (u, v) in nonlethal3_moves_own:
                score += weights['W_NonLethal3']
            if (u, v) in nonlethal3_moves_opp:
                score += weights['W_NonLethal3_Block']
            
            if (u, v) in type2_moves_own:
                score += weights['W_Type2']
            if (u, v) in type2_moves_opp:
                score += weights['W_Type2_Block']

            if (u, v) in weaktype2_moves_own:
                score += weights['W_WeakType2']
            if (u, v) in weaktype2_moves_opp:
                score += weights['W_WeakType2_Block'] 
            
            # Surrounding/proximity bonus: prefer moves adjacent to your own stones
            if detector.has_nhd(board, u, v):
                score += weights['W_Surround']

            scores[(u, v)] = score
    
    # store in cache if possible
    try:
        if key is not None:
            _raw_cache[key] = dict(scores)
            if len(_raw_cache) > _CACHE_SIZE:
                _raw_cache.popitem(last=False)
    except Exception:
        pass
    return scores

    # unreachable


def kernel_scores(board, color, candidates=None, sigma=1.0, alpha=0.25):
    '''
    Compute propagated kernel-style scores for each playable cell.

    This implementation uses additive propagation instead of averaging: the
    propagated score at point p is raw_score(p) + alpha * sum(raw_score(neighbor)).

    Parameters:
    - board, color: as usual
    - sigma: preserved for API compatibility but not used for propagation here

    Returns:
    - dict mapping (u,v) -> propagated_score (normalized over candidates)
    '''
    # try module-level cache first (kernel scores depend only on board+color)
    try:
        # include id(raw_scores) so tests that monkeypatch raw_scores get fresh results
        kkey = (board.tobytes(), int(color), id(raw_scores))
    except Exception:
        kkey = None
    if kkey is not None:
        cached = _kernel_cache.get(kkey)
        if cached is not None:
            return dict(cached)

    # compute raw scores
    raw = raw_scores(board, color)

    # if no explicit candidate set passed, try to obtain one from detector
    if candidates is None:
        try:
            cand_res = detector.candidates_moves(board, color)
            # detector.candidate_moves may return (candidates, priority)
            if isinstance(cand_res, (tuple, list)) and len(cand_res) > 0:
                cand = cand_res[0]
            else:
                cand = cand_res
            candidates = set(cand) if cand is not None else set()
        except Exception:
            candidates = set()
    else:
        # make a defensive copy / convert to set
        try:
            candidates = set(candidates)
        except Exception:
            candidates = set()

    n = board.shape[0]
    propagated = {}

    # neighbor offsets on our triangular grid (6-directional)
    neigh_offsets = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, -1), (-1, 1)]

    for u in range(n):
        for v in range(n - u):
            if not detector._is_valid(board, u, v):
                continue

            base = raw.get((u, v), 0.0)
            nbr_sum = 0.0
            for du, dv in neigh_offsets:
                nu, nv = u + du, v + dv
                if (nu, nv) in raw:
                    nbr_sum += raw[(nu, nv)]

            propagated[(u, v)] = base + alpha * nbr_sum

    # Normalize propagated scores over all playable cells (uniform scale).
    keys = list(propagated.keys())
    if keys:
        vals = np.array([propagated[k] for k in keys], dtype=float)
        mn = float(np.min(vals))
        mx = float(np.max(vals))
        if mx > mn + 1e-12:
            rng = mx - mn
            out = {k: float((v - mn) / rng) for k, v in propagated.items()}
            try:
                if kkey is not None:
                    _kernel_cache[kkey] = dict(out)
                    if len(_kernel_cache) > _CACHE_SIZE:
                        _kernel_cache.popitem(last=False)
            except Exception:
                pass
            return out
        else:
            # all equal -> set all normalized values to 0.0
            out = {k: 0.0 for k in propagated.keys()}
            try:
                if kkey is not None:
                    _kernel_cache[kkey] = dict(out)
                    if len(_kernel_cache) > _CACHE_SIZE:
                        _kernel_cache.popitem(last=False)
            except Exception:
                pass
            return out

    try:
        if kkey is not None:
            _kernel_cache[kkey] = dict(propagated)
            if len(_kernel_cache) > _CACHE_SIZE:
                _kernel_cache.popitem(last=False)
    except Exception:
        pass
    return propagated

