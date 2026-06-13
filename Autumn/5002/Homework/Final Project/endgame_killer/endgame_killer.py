''' Module that integrates all strategy layers for final endgame killer AI '''

import numpy as np
import detector
import score
import MCTS
from tree import Node
from typing import Optional
import importlib
import time

# Telemetry counters for VCT vs detector candidate overlap analysis
_telemetry = {
    'vct_runs': 0,
    'vct_found': 0,
    'vct_found_in_candidates': 0,
    'vct_found_not_in_candidates': 0,
    'opp_vct_runs': 0,
    'opp_vct_found': 0,
    'opp_vct_found_in_candidates': 0,
    'opp_vct_found_not_in_candidates': 0,
}

def get_telemetry():
    """Return a shallow copy of telemetry counters."""
    return dict(_telemetry)

def reset_telemetry():
    """Reset telemetry counters to zero."""
    for k in _telemetry:
        _telemetry[k] = 0


def random_move(board, color):
    '''Random move for the final fallback case'''
    empties = list(zip(*np.where(board == 0)))
    if not empties:
        return None
    return empties[np.random.randint(len(empties))]


def score_based_AI(board, color, debug=False):
    ''' Simple AI with only helper functions in detector and score modules.'''
    own = color # own color
    opp = -color # opponent color
    candidates, level = detector.candidates_moves(board, color)
    
    # Null board for 1st hand play when AI is black
    nullboard = np.zeros((15,15), dtype=int)
    nullboard[np.triu_indices(15, k=1)] = 5
    nullboard = np.flipud(nullboard)
    
    # If first move as Black, choose center
    if own == 1 and np.array_equal(board, nullboard):
        if debug:
            print("endgame_killer: first move, choosing center (6,4)")
        return (6,4)

    # If we have high-priority tactical candidates, choose deterministically
    # without computing expensive scoring. Use proximity-to-center as tie-breaker.
    elif candidates:
        # levels: 1=own immediate win, 2=opp immediate win, 3=own lethal type3, 4=opp lethal type3
        if level in (1, 2, 3, 4):
            best = min(candidates, key=lambda mv: (mv[0]-6)**2 + (mv[1]-4)**2)
            if debug:
                print("endgame_killer: candidates=", candidates, "chosen=", best, "level=", level)
            return best

        # fallback: compute kernel scores and pick best
        own_scores = score.kernel_scores(board, own)
        # deterministic tie-break: prefer moves with higher score, then closer to center (6,4)
        center_u, center_v = 6, 4
        def fallback_key(mv):
            sc = own_scores.get(mv, float('-inf'))
            dist = (mv[0]-center_u)**2 + (mv[1]-center_v)**2
            return (sc, -dist)

        best = max(candidates, key=fallback_key)
        if debug:
            print("endgame_killer: No lethal threats detected", "chosen=", best)
        return best
    else:
        mv = random_move(board, color)
        if debug:
            print("endgame_killer: random fallback", mv)
        return mv


def MCTS_score_AI(board, color, iterations: int = 200, c_puct: float = 1.0,
                  debug: bool = False, time_budget: Optional[float] = 5.0,
                  top_k: Optional[int] = None, center_first_hand: bool = True):
    """Run MCTS seeded by detector candidates and using kernel_scores for leaf eval.

    Parameters:
    - `iterations`: iteration-based stop condition (used only when
      `time_budget` is None).
    - `time_budget`: if not None, MCTS will run until this many seconds
      have elapsed (wall-clock). When `time_budget` is provided it
      *takes precedence* over `iterations` — the search will keep
      running until the time budget is exhausted, and `iterations` is
      ignored.

    Returns a move tuple (u,v) or None.
    """

    if center_first_hand:
        # Null board for 1st hand play when AI is black. Honor this shortcut
        # only when `center_first_hand` is True — tuning scripts can disable it
        # to let MCTS explore the opening freely.
        nullboard = np.zeros((15,15), dtype=int)
        nullboard[np.triu_indices(15, k=1)] = 5
        nullboard = np.flipud(nullboard)

        # If first move as Black, choose center
        if color == 1 and np.array_equal(board, nullboard):
            if debug:
                print("endgame_killer: first move, choosing center (6,4)")
            return (6,4)
    

    # get detector candidates and quick priority
    candidates, level = detector.candidates_moves(board, color)

    # Shallow opponent VCT: quickly check whether opponent (-color) has a
    # forced win. If so, play the opponent's first PV move now to block it.
    # This is a shallow, time-limited VCT meant to catch forcing sequences
    # (including situations that would previously be called "double-threats")
    # without always running a deep proof search.
    try:
        # avoid checking if we have an immediate winning move ourselves
        if level != 1:
            VCT_mod = None
            try:
                import VCT as VCT_mod
                importlib.reload(VCT_mod)
            except Exception:
                VCT_mod = None

            if VCT_mod is not None and hasattr(VCT_mod, 'find_vct'):
                # allocate a small shallow budget for opponent-proof search
                if time_budget is not None:
                    opp_budget = min(0.25, max(0.01, float(time_budget) * 0.15))
                else:
                    opp_budget = 0.25
                _telemetry['opp_vct_runs'] += 1
                pv_opp = VCT_mod.find_vct(board, -color, max_depth=6, top_k=top_k if top_k is not None else 6, debug=debug, max_time=opp_budget)
                if pv_opp:
                    _telemetry['opp_vct_found'] += 1
                    blk = pv_opp[0]
                    # record whether opponent PV first move was already in detector candidates
                    try:
                        if blk in candidates:
                            _telemetry['opp_vct_found_in_candidates'] += 1
                        else:
                            _telemetry['opp_vct_found_not_in_candidates'] += 1
                    except Exception:
                        # conservative: if candidates isn't iterable, treat as not-in
                        _telemetry['opp_vct_found_not_in_candidates'] += 1

                    # pv_opp[0] is the opponent's planned first move; block it if legal
                    if detector._is_valid(board, blk[0], blk[1]):
                        if debug:
                            print(f"MCTS_score_AI: opponent VCT found PV, blocking move {blk} (elapsed budget {opp_budget}s)")
                        return blk
    except Exception:
        # ignore VCT failures and continue with normal flow
        pass

    # Quick VCT check: if a forcing sequence exists for `color`, return its first move.
    # Use `top_k` fallback if provided, otherwise let VCT use its default.
    # Quick VCT check: run when any threat exists (type2/type3/type4/double).
    # Use `detector.if_exist_threats` so non-lethal threats (type2/weak-type2)
    # also trigger a VCT attempt without polluting `candidates_moves`.
    try:
        threat_exists = detector.if_exist_threats(board, color)
    except Exception:
        threat_exists = (level != 99)

    if threat_exists:
        try:
            # Ensure `detector.check_winner` exists in the running module; reload if necessary.
            if not hasattr(detector, 'check_winner'):
                importlib.reload(detector)

            VCT_mod = None
            try:
                import VCT as VCT_mod
                # reload to pick up recent edits in interactive sessions
                importlib.reload(VCT_mod)
            except Exception:
                VCT_mod = None

            if VCT_mod is not None and hasattr(VCT_mod, 'find_vct'):
                # limit VCT runtime so it cannot blow past the caller's time_budget
                if time_budget is not None:
                    # allocate at most 25% of overall budget to VCT, cap at 0.5s
                    vct_budget = min(0.5, max(0.01, float(time_budget) * 0.25))
                else:
                    # when no global budget is given, use a conservative cap
                    vct_budget = 0.5
                # telemetry: time the VCT run (only when debug is True)
                t0 = time.time()
                _telemetry['vct_runs'] += 1
                pv = VCT_mod.find_vct(board, color, max_depth=12, top_k=top_k if top_k is not None else 8, debug=debug, max_time=vct_budget)
                t1 = time.time()
                if debug:
                    found = bool(pv)
                    pv_len = len(pv) if pv else 0
                    print(f"MCTS_score_AI: VCT run finished (found={found}, pv_len={pv_len}, elapsed={t1-t0:.3f}s, vct_budget={vct_budget:.3f}s)")
                if pv:
                    _telemetry['vct_found'] += 1
                    try:
                        if pv[0] in candidates:
                            _telemetry['vct_found_in_candidates'] += 1
                        else:
                            _telemetry['vct_found_not_in_candidates'] += 1
                    except Exception:
                        _telemetry['vct_found_not_in_candidates'] += 1

                    if debug:
                        print('MCTS_score_AI: VCT found forcing PV, returning first move', pv[0])
                    return pv[0]
        except Exception:
            # If VCT or detector reload fails for any reason, continue with MCTS as fallback.
            pass

    # If detector returns a single forced candidate, return it immediately.
    # This avoids running MCTS for trivial forced moves and keeps behavior
    # consistent with the detector-based short-circuits used elsewhere.
    if candidates:
        try:
            if len(candidates) == 1:
                only = next(iter(candidates))
                if debug:
                    print('MCTS_score_AI: single detector candidate, returning', only)
                return only
        except Exception:
            # If `candidates` isn't sized/iterable in the expected way,
            # ignore and continue to MCTS as a safe fallback.
            pass

    # Build simple callbacks over the numpy board representation
    def generate_moves(pos):
        n = pos.shape[0]
        for u in range(n):
            for v in range(n - u):
                if detector._is_valid(pos, u, v):
                    yield (u, v)

    def side_to_move(pos):
        # black (1) plays when counts equal, otherwise white (-1)
        blacks = int(np.sum(pos == 1))
        whites = int(np.sum(pos == -1))
        return 1 if blacks == whites else -1

    def make_move(pos, mv):
        new = pos.copy()
        to_play = side_to_move(pos)
        new[mv[0], mv[1]] = to_play
        return new

    root = Node()

    # If detector returned a non-empty candidate set, pass it to MCTS to focus the root.
    root_cands = set(candidates) if candidates is not None else None

    # Provide a small deterministic evaluator for alpha-beta that mirrors the
    # kernel-score-based fallback used elsewhere. This evaluator returns a
    # value in [-1,1] from the viewpoint of the side to move: positive means
    # advantage to the side to move.
    def evaluate(pos):
        try:
            to_move = side_to_move(pos)
            # use a small per-search cache for kernel-score maxima to avoid
            # recomputing expensive propagated scores repeatedly on similar
            # positions during MCTS.
            if not hasattr(evaluate, '_ks_max_cache'):
                evaluate._ks_max_cache = {}
            cache = evaluate._ks_max_cache
            key_own = (pos.tobytes(), int(to_move))
            key_opp = (pos.tobytes(), int(-to_move))

            if key_own in cache:
                max_own = cache[key_own]
            else:
                ks_own = score.kernel_scores(pos, to_move)
                max_own = float(max(ks_own.values())) if ks_own else 0.0
                cache[key_own] = max_own

            if key_opp in cache:
                max_opp = cache[key_opp]
            else:
                ks_opp = score.kernel_scores(pos, -to_move)
                max_opp = float(max(ks_opp.values())) if ks_opp else 0.0
                cache[key_opp] = max_opp

            return float(max_own - max_opp)
        except Exception:
            return 0.0

    if debug:
        print(f"MCTS_score_AI: running MCTS with iterations={iterations}, time_budget={time_budget}, c_puct={c_puct}, top_k={top_k}, detector_level={level}")
    MCTS.mcts_search(root, board, generate_moves, make_move, evaluate, side_to_move,
                     iterations=iterations, c_puct=c_puct, leaf_depth=3,
                     root_candidates=root_cands, restrict_to_candidates=False,
                     use_kernel_score=True, time_budget=time_budget, top_k=top_k, debug=debug)

    best = MCTS.best_move_by_visits(root)
    if debug:
        print('MCTS_score_AI: detector level=', level, 'candidates=', len(candidates) if candidates else 0, 'best=', best)
    return best

def endgame_killer(board, color, debug=False):
    return MCTS_score_AI(board, color, c_puct=0.8, top_k=3, time_budget=2, debug=True)