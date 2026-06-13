"""VCT (Victory by Continuous Threats) helper.

This module implements a small, depth-limited proof-search for forcing
winning sequences (exists-forall-exists ...). It is intentionally
conservative and focused on using the existing `detector.candidates_moves`
as the primary generator of forcing moves. When the detector returns no
high-priority candidates, the search falls back to a small `top_k` set
selected by `score.kernel_scores`.

The main exported function is `find_vct(board, color, max_depth, top_k)`.
It returns a list of moves (principal variation) when a forced win for
`color` is found within `max_depth` plies, otherwise `None`.

This is not a full-featured VCT/VCF engine, but it's useful as a tactical
solver for short forcing sequences and integrates cleanly with the
existing detector/score APIs.
"""

from typing import List, Optional, Tuple, Dict
import numpy as np
import detector
import score
import time


def _make_move(board: np.ndarray, mv: Tuple[int, int], color: int) -> np.ndarray:
	new = board.copy()
	new[mv[0], mv[1]] = color
	return new


def _legal_moves_from_detector(board: np.ndarray, color: int, top_k: Optional[int]) -> Tuple[List[Tuple[int,int]], int]:
	"""Return a list of candidate moves and detector level.

	If the detector returns a tactical level (1..6), return those moves.
	Otherwise, fall back to the top_k moves by kernel score.
	"""
	cands, level = detector.candidates_moves(board, color)
	if cands:
		# detector may return a set-like object
		try:
			moves = list(cands)
		except Exception:
			moves = [c for c in cands]

		# Augment detector moves with weaker tactical signals that can
		# combine into stronger threats (e.g. two type-2s creating a 33).
		try:
			# include own type-2 moves and weak-type-2 moves
			t2 = detector.type2_moves(board, color)
			wt2 = detector.weak_type2_moves(board, color)
		except Exception:
			# if detector does not provide these, ignore
			t2 = set(); wt2 = set()
		moves_set = set(moves) | set(t2) | set(wt2)
		moves = list(moves_set)

		# When detector gives many fallback moves (level==99) or the
		# augmented set is large, trim to `top_k` using kernel scores.
		if top_k is not None and len(moves) > top_k:
			ks = score.kernel_scores(board, color)
			moves = sorted(moves, key=lambda m: ks.get(m, 0.0), reverse=True)[:top_k]
		return moves, level

	# detector returned no candidates: use kernel_scores top_k over empties
	empties = list(zip(*np.where(board == 0)))
	if not empties:
		return [], 99
	ks = score.kernel_scores(board, color)
	if top_k is None:
		return list(empties), 99
	moves_sorted = sorted(empties, key=lambda m: ks.get(m, 0.0), reverse=True)
	return moves_sorted[:top_k], 99


def find_vct(board: np.ndarray, color: int, max_depth: int = 12, top_k: int = 8, debug: bool = False, max_time: Optional[float] = None) -> Optional[List[Tuple[int,int]]]:
	"""Attempt to find a forced win (VCT) for `color` within `max_depth` plies.

	Returns a PV list of moves (from root) when found, otherwise None.
	"""
	# transposition / memo cache: key -> (proven_bool, pv or None)
	cache: Dict[Tuple[bytes, int, int], Optional[List[Tuple[int,int]]]] = {}

	start_time = time.time()
	end_time = start_time + max_time if max_time is not None else None

	def dfs(pos: np.ndarray, to_move: int, depth: int) -> Optional[List[Tuple[int,int]]]:
		# check terminal using detector.check_winner to avoid circular imports
		winner, _ = detector.check_winner(pos)
		if winner != 0:
			# winner==color means previous move gave color a win
			return [] if winner == color else None
		# time cutoff: if we've used up our allotted VCT time, abort search
		if end_time is not None and time.time() >= end_time:
			return None
		if depth <= 0:
			return None

		key = (pos.tobytes(), int(to_move), depth)
		if key in cache:
			return cache[key]

		# pick candidate moves for the side to move
		moves, level = _legal_moves_from_detector(pos, to_move, top_k)
		if not moves:
			cache[key] = None
			return None

		if to_move == color:
			# Existential node: we need one move such that for all opponent replies
			# the continuation proves a win.
			for mv in moves:
				new = _make_move(pos, mv, to_move)
				# immediate win?
				w, _ = detector.check_winner(new)
				if w == color:
					cache[key] = [mv]
					if debug:
						print(f"VCT: winning move found for {color}: {mv} at depth {depth}")
					return [mv]

				# Now opponent to move: we require that for ALL opponent replies,
				# the recursive call returns a forced win for `color`.
				opp_moves, _ = _legal_moves_from_detector(new, -to_move, top_k)
				if not opp_moves:
					# opponent has no reply -> treat as win
					cache[key] = [mv]
					return [mv]

				all_responses_ok = True
				# For each possible opponent response, the continuation must be proven
				for opp_mv in opp_moves:
					new2 = _make_move(new, opp_mv, -to_move)
					res = dfs(new2, to_move, depth - 2)
					if res is None:
						all_responses_ok = False
						break
				if all_responses_ok:
					# prepend current mv to PV from first child (they can differ but all lead to win)
					# we return the move sequence starting with mv
					cache[key] = [mv]
					return [mv]
			cache[key] = None
			return None
		else:
			# Universal node for opponent: if opponent has any move that avoids
			# the forced win, then this node fails (i.e., return None).
			opp_moves, _ = moves, level
			for mv in opp_moves:
				new = _make_move(pos, mv, to_move)
				# if opponent move leads to immediate opponent win, then failure
				w, _ = detector.check_winner(new)
				if w == -color:
					cache[key] = None
					return None
				res = dfs(new, -to_move, depth - 1)
				if res is None:
					# opponent found a reply that escapes
					cache[key] = None
					return None
			# if opponent has no escaping move, then from the POV of the previous
			# existential node this path is winning; return empty PV (no move to add here)
			cache[key] = []
			return []

	pv = dfs(board, color, max_depth)
	if pv:
		return pv
	return None


if __name__ == '__main__':
	# quick smoke test on the nullboard
	b = np.zeros((15,15), dtype=int)
	b[np.triu_indices(15, k=1)] = 5
	b = np.flipud(b)
	print('VCT find on nullboard (expected None):', find_vct(b, 1, max_depth=8, top_k=6))

