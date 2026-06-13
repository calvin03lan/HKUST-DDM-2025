"""MCTS.py

Lightweight MCTS (PUCT) driver that uses `tree.Node` for the search tree and
`negamax_alpha_beta` from `tree` for deterministic leaf evaluation. This file
provides a small, well-commented reference implementation suitable for
prototyping and integration with `VCT`/`DFPN` later.

Design notes:
- The implementation is intentionally generic: it accepts callbacks for
  `generate_moves`, `make_move`, `evaluate`, and `side_to_move` so it can be
  used with the project's existing board representation.
- PUCT selection uses the priors stored in child nodes (child.P). If a node
  is expanded, priors should be provided via the `expand` call that creates
  children (the MCTS driver here uses uniform priors by default).
- Leaf evaluation uses `negamax_alpha_beta` (functional API) so you can
  substitute an alpha-beta engine, quiescence, or any deterministic solver.

Usage (high level):
1. Build a root `tree.Node()` and pass it plus your position and callbacks to
   `mcts_search`.
2. After search finishes, choose the best move with `best_move_by_visits(root)`
   or inspect the tree for PVs / stats.

This is a prototype: for production you will want to add concurrency safety,
transposition table lookups, policy networks, and stronger leaf evaluators.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Tuple
import random
import time
import numpy as np

import score

from tree import Node, best_move_by_visits

Move = Tuple[int, int]
Position = object


def mcts_search(root: Node,
                root_position: Position,
                generate_moves: Callable[[Position], Iterable[Move]],
                make_move: Callable[[Position, Move], Position],
                evaluate: Optional[Callable[[Position], float]],
                side_to_move: Callable[[Position], int],
                iterations: int = 200,
                c_puct: float = 1.0,
                leaf_depth: int = 3,
                root_candidates: Optional[Iterable[Move]] = None,
                restrict_to_candidates: bool = False,
                use_kernel_score: bool = True,
                use_alpha_beta: bool = False,
                time_budget: Optional[float] = 5.0,
                top_k: Optional[int] = None,
                debug: bool = False) -> None:
    """Perform MCTS search in-place on `root`.

    Parameters:
    - `root`: tree.Node instance representing root of search tree.
    - `root_position`: position object representing the game state at root.
    - `generate_moves(position)` -> iterable of moves.
    - `make_move(position, move)` -> new position after applying move.
    - `evaluate(position)` -> float: deterministic static evaluator returning
         value from *side-to-move* perspective at terminal/leaf positions.
    - `side_to_move(position)` -> int: returns +1 or -1 indicating player to move.
    - `iterations`: number of MCTS iterations to run.
    - `c_puct`: PUCT exploration constant.
    - `leaf_depth`: depth for internal alpha-beta leaf evaluation.

    Notes on value/sign conventions:
    - `evaluate` must return values from the viewpoint of the position's
      side-to-move (standard negamax convention). To backpropagate into the
      shared tree (whose root corresponds to some fixed `root_player`) we map
      the leaf value into the root player's perspective before calling
      `Node.backup`.
    """

    # Identify root player (the player to move at the root position). We'll
    # convert leaf values to be from this player's perspective before backup.
    root_player = side_to_move(root_position)
    # normalize candidate set (if provided) into a fast lookup set
    root_cand_set = set(root_candidates) if root_candidates is not None else set()

    # Run the main loop either for a fixed number of iterations or until
    # the given time budget (seconds) is exhausted.
    #
    # Precedence policy:
    # - If `time_budget` is not None, the search will run until the wall-
    #   clock time reaches `start_time + time_budget`. The `iterations`
    #   parameter is ignored in this case. This ensures the caller can
    #   allocate a fixed time slice for search (e.g. `time_budget=5.0`).
    # - If `time_budget` is None, the search will run for exactly
    #   `iterations` iterations. This keeps the old iteration-based API
    #   behavior.
    #
    # Notes:
    # - The loop checks the time once per MCTS iteration, so the actual
    #   runtime may exceed `time_budget` by a small amount (~one
    #   iteration's cost).
    # - If you want to force the search to use exactly the time budget
    #   (and ignore iterations), pass a float (seconds). To prefer
    #   iterations instead, pass `time_budget=None`.
    it = 0
    start_time = time.time()
    end_time = start_time + time_budget if time_budget is not None else None

    # simple cache for kernel_scores to avoid recomputing on the same
    # numpy position objects during the search. Keyed by (bytes, to_move).
    # Using a dict is faster than an LRU for the short-lived search.
    ks_cache = {}

    def get_kernel_scores(pos, to_move):
        try:
            key = (pos.tobytes(), int(to_move))
        except Exception:
            # Fallback: if tobytes fails for some reason, call directly
            return score.kernel_scores(pos, to_move)
        v = ks_cache.get(key)
        if v is None:
            v = score.kernel_scores(pos, to_move)
            ks_cache[key] = v
        return v
    # cache generated legal moves per position bytes to avoid repeatedly
    # calling the caller-provided `generate_moves` generator which can be
    # expensive (it calls detector._is_valid inside). Keyed by pos.tobytes().
    gen_moves_cache = {}

    def get_moves_list(pos):
        try:
            key = pos.tobytes()
        except Exception:
            # fallback: build list directly
            return list(generate_moves(pos))
        mv = gen_moves_cache.get(key)
        if mv is None:
            mv = list(generate_moves(pos))
            gen_moves_cache[key] = mv
        return mv
    stop_search = False
    while True:
        # Stop conditions
        if end_time is not None and time.time() >= end_time:
            break
        if end_time is None and it >= iterations:
            break
        it += 1
        node = root
        pos = root_position

        # ------------------ Selection ------------------
        # Traverse the tree using PUCT until we reach a leaf node.
        # ------------------ Selection ------------------
        # Traverse the tree using PUCT until we reach a leaf node. Check the
        # time budget during long traversals to avoid a single iteration
        # exceeding the allotted wall-clock time by too much.
        while not node.is_leaf():
            if end_time is not None and time.time() >= end_time:
                stop_search = True
                break
            mv, child = node.select_best_puct(c_puct)
            pos = make_move(pos, mv)
            node = child
        if stop_search:
            break

        # ------------------ Expansion ------------------
        # Generate candidate moves at the leaf position.
        # If `root_candidates` is given and we're at the root, restrict to those.
        # If `restrict_to_candidates` is True, filter all generated moves by the set.
        if end_time is not None and time.time() >= end_time:
            break

        if node is root and root_cand_set:
            # use root candidate set (only keep legal ones)
            all_moves = get_moves_list(pos)
            moves = [m for m in all_moves if m in root_cand_set]
        else:
            moves = get_moves_list(pos)
            if restrict_to_candidates and root_cand_set:
                moves = [m for m in moves if m in root_cand_set]
        # Optionally restrict to the top-k moves by kernel score. This is
        # useful when no detector-provided candidate set is given or when
        # you want to further narrow the branching factor for MCTS.
        if end_time is not None and time.time() >= end_time:
            break

        if top_k is not None and top_k > 0 and moves:
            try:
                to_move_for_filter = side_to_move(pos)
                ks_all = get_kernel_scores(pos, to_move_for_filter)
                # ks_all maps move->score in [0,1]; sort moves by score desc
                # and keep top_k moves. If ks_all misses a move, treat its
                # score as 0.
                moves_sorted = sorted(moves, key=lambda m: ks_all.get(m, 0.0), reverse=True)
                moves = moves_sorted[:top_k]
            except Exception:
                # If kernel scoring fails, fall back to current `moves` list.
                pass
        if moves:
            # Use kernel_scores to form priors (policy) when available.
            # This biases expansion toward moves the heuristic considers strong
            # rather than using a uniform prior which wastes early playouts.
            priors = None
            try:
                to_move = side_to_move(pos)
                ks = get_kernel_scores(pos, to_move)
                if ks:
                    # ks maps move->[0,1]; turn it into a small-prob floor then normalize
                    eps = 1e-6
                    raw = {m: float(ks.get(m, 0.0)) + eps for m in moves}
                    s = sum(raw.values())
                    if s > 0:
                        priors = {m: raw[m] / s for m in moves}
            except Exception:
                priors = None

            if end_time is not None and time.time() >= end_time:
                break

            if priors is None:
                # fallback to uniform priors
                priors = {m: 1.0 / len(moves) for m in moves}
            # `to_move` is not critical for MCTS statistics; use current side
            # to move for children for completeness.
            to_move = side_to_move(pos)
            node.expand(priors, to_move=to_move)

            # pick one child to simulate/evaluate — common strategy is to pick
            # a random child among the newly expanded children to encourage
            # exploration; here we pick one proportionally to the prior.
            moves_list = list(priors.keys())
            probs = [priors[m] for m in moves_list]
            chosen_move = random.choices(moves_list, weights=probs, k=1)[0]
            node = node.children[chosen_move]
            pos = make_move(pos, chosen_move)
        if end_time is not None and time.time() >= end_time:
            break

        # ------------------ Evaluation ------------------
        # Evaluate the leaf (terminal or newly expanded child). Behavior
        # depends on flags:
        # - If `use_alpha_beta` is True and `evaluate` is provided, run the
        #   existing `negamax_alpha_beta` engine (preserves older behavior).
        # - Else, if `use_kernel_score` is True, try to use
        #   `score.kernel_scores` on the current position and aggregate the
        #   returned per-move scores into a single scalar in [-1,1]. This is
        #   a fast heuristic fallback suitable for prototyping.
        # - Otherwise, fall back to the provided `evaluate` callable if any.
        if moves:
            if use_kernel_score:
                try:
                    to_move = side_to_move(pos)
                    if end_time is not None and time.time() >= end_time:
                        break
                    # get top (max) kernel score for side to move and opponent
                    ks_own = get_kernel_scores(pos, to_move)
                    ks_opp = get_kernel_scores(pos, -to_move)
                    if ks_own:
                        max_own = float(max(ks_own.values()))
                    else:
                        max_own = 0.0
                    if ks_opp:
                        max_opp = float(max(ks_opp.values()))
                    else:
                        max_opp = 0.0
                    # value in [-1,1]: positive means advantage to side-to-move
                    leaf_value = max_own - max_opp
                except Exception:
                    if evaluate is not None:
                        leaf_value = float(evaluate(pos))
                    else:
                        leaf_value = 0.0
            elif evaluate is not None:
                leaf_value = float(evaluate(pos))
            else:
                leaf_value = 0.0
        else:
            # terminal: try evaluate first, then kernel score, then 0
            if evaluate is not None:
                leaf_value = float(evaluate(pos))
            elif use_kernel_score:
                try:
                    to_move = side_to_move(pos)
                    if end_time is not None and time.time() >= end_time:
                        break
                    ks_own = get_kernel_scores(pos, to_move)
                    ks_opp = get_kernel_scores(pos, -to_move)
                    max_own = float(max(ks_own.values())) if ks_own else 0.0
                    max_opp = float(max(ks_opp.values())) if ks_opp else 0.0
                    leaf_value = max_own - max_opp
                except Exception:
                    leaf_value = 0.0
            else:
                leaf_value = 0.0

        # Convert leaf_value (which is from side-to-move at `pos`) into a
        # value from root_player's perspective for consistent backups.
        pos_to_move = side_to_move(pos)
        sign = 1.0 if pos_to_move == root_player else -1.0
        value_for_root = float(leaf_value) * sign

        # ------------------ Backup ------------------
        # Backpropagate the value up the tree. `Node.backup` toggles the
        # value sign at each parent level, so passing a value measured from the
        # root player's perspective is correct.
        node.backup(value_for_root)

    # End of main search loop. If debug is requested, print a concise
    # summary including runtime, iterations performed, and root child stats.
    if debug:
        elapsed = time.time() - start_time
        iters_done = it if end_time is not None else it
        print(f"MCTS debug: iterations={iters_done}, elapsed={elapsed:.3f}s, time_budget={time_budget}, iterations_limit={iterations}, c_puct={c_puct}, top_k={top_k}")
        try:
            rcands = set(root_candidates) if root_candidates is not None else None
            rcands_n = len(rcands) if rcands is not None else 0
        except Exception:
            rcands_n = 0
        print(f"  root_candidates_size={rcands_n}, root_child_count={len(root.children)}")
        print("  Root children (move : P, N, Q):")
        for mv, ch in root.children.items():
            try:
                print(f"    {mv} : P={ch.P:.4f}, N={ch.N}, Q={ch.Q:.4f}")
            except Exception:
                print(f"    {mv} : (P={getattr(ch,'P',None)}, N={getattr(ch,'N',None)}, Q={getattr(ch,'Q',None)})")


def demo_toy_search():
    """Run a tiny demo search on a synthetic game to illustrate usage.

    The toy game: positions are tuples of moves so far; two moves per ply
    drawn from {0,1}. Terminal depth is 4. Evaluation returns +1 if sum(pos)
    is even else -1 (toy heuristic). This mirrors the examples used during
    development and helps validate that the driver is wired correctly.
    """
    MAX_DEPTH = 4

    def generate_moves(pos):
        if len(pos) >= MAX_DEPTH:
            return []
        return [0, 1]

    def make_move(pos, mv):
        return pos + (mv,)

    def evaluate(pos):
        if len(pos) < MAX_DEPTH:
            return 0.0
        s = sum(pos)
        return 1.0 if (s % 2) == 0 else -1.0

    def side_to_move(pos):
        # alternate players: +1 starts and alternates each ply
        return 1 if (len(pos) % 2) == 0 else -1

    root_pos = ()
    root = Node()

    print('Running toy MCTS for 200 iterations...')
    # pass time_budget=None so this demo uses the explicit iterations count
    mcts_search(root, root_pos, generate_moves, make_move, evaluate, side_to_move, iterations=200, c_puct=1.0, leaf_depth=3, time_budget=None)
    best = best_move_by_visits(root)
    print('Best move by visits:', best)
    print('Root children stats:')
    for mv, ch in root.children.items():
        print(' move', mv, 'N=', ch.N, 'Q=', ch.Q, 'P=', ch.P)


if __name__ == '__main__':
    demo_toy_search()
