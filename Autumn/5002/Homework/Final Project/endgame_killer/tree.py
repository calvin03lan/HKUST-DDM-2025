"""
tree.py

Lightweight tree/node utilities intended to be shared by MCTS/alpha-beta, VCT/VCF and DFPN.

Design goals:
- Minimal memory footprint: use __slots__ on Node to reduce Python object overhead.
- Simple API: Node stores common fields used by MCTS (N/W/Q/P) and DFPN (pn/dn, proven).
- Small set of helper functions for selection, backup, PV extraction and diagnostics.
- Extensive docstrings and inline comments so the module is easy to reuse and extend.

Usage sketch:
- MCTS: create root Node, call expand() with priors, use select_best_uct and backup.
- VCT/VCF: build a small tree from forcing moves, use prune_children to limit branching,
  and get_pv() to extract the forcing line when a proven/solved node is found.
- DFPN: reuse pn/dn and mark_proven fields; a DFPN driver can store pn/dn in the node
  and call propagate functions (not implemented here) while using the Node shape.

This file intentionally stays implementation-light for DFPN internals (pn/dn
propagation policies are search-specific). It provides the node structure and
common operations that are helpful across search algorithms.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, List, Iterable
import math

Move = Tuple[int, int]


class Node:
    """A compact tree node suitable for MCTS / VCT / DFPN.

    Fields:
    - parent: reference to parent Node or None for root
    - move: the move (u,v) that led to this node from parent (None for root)
    - to_move: which color is to move at this node (1 or -1)
    - P: prior probability (float) used by PUCT/PUCT-like selection
    - N: visit count (int)
    - W: total value (float) accumulated from backups (signed from root perspective)
    - Q: mean value (float) == W/N when N>0 else 0.0
    - children: dict mapping Move -> Node

    DFPN/VCF helpers:
    - pn, dn: proof/disproof numbers (floats). Default 1.0. Algorithms may update these.
    - proven: None or +1/-1 indicating a proven win/loss for the side to move at node.

    Implementation notes:
    - Use __slots__ to keep instances small (important when many nodes are created).
    - Methods are intentionally minimal; higher-level search drivers implement
      algorithm-specific logic (e.g., DFPN propagation).
    """

    __slots__ = (
        "parent",
        "move",
        "to_move",
        "P",
        "N",
        "W",
        "Q",
        "children",
        "pn",
        "dn",
        "proven",
    )

    def __init__(self, parent: Optional["Node"] = None, move: Optional[Move] = None,
                 to_move: int = 1, prior: float = 0.0):
        # tree structure
        self.parent: Optional[Node] = parent
        self.move: Optional[Move] = move
        self.to_move: int = int(to_move)

        # MCTS / PUCT fields
        self.P: float = float(prior)  # prior probability from policy/prior function
        self.N: int = 0               # visit count
        self.W: float = 0.0           # total value
        self.Q: float = 0.0           # mean value (W/N)

        # children mapping: Move -> Node
        self.children: Dict[Move, Node] = {}

        # DFPN fields (optional)
        self.pn: float = 1.0
        self.dn: float = 1.0
        self.proven: Optional[int] = None

    # -------------------------- basic tree ops --------------------------
    def is_leaf(self) -> bool:
        """Return True if node has no children.

        This is used by MCTS selection and by VCT/DFPN drivers to detect
        expansion points.
        """
        return len(self.children) == 0

    def add_child(self, move: Move, to_move: int, prior: float = 0.0) -> "Node":
        """Add (or return existing) child node for `move`.

        Returns the child Node instance. This is idempotent when called multiple times
        for the same move.
        """
        if move in self.children:
            return self.children[move]
        child = Node(parent=self, move=move, to_move=to_move, prior=prior)
        self.children[move] = child
        return child

    def expand(self, priors: Dict[Move, float], to_move: int):
        """Bulk-create children from a `priors` dict mapping move->prior.

        Existing children are preserved; new moves are added with given prior.
        `to_move` is stored on newly created child nodes.
        """
        for m, p in priors.items():
            if m not in self.children:
                self.children[m] = Node(parent=self, move=m, to_move=to_move, prior=float(p))

    # ---------------------- MCTS / PUCT selection -----------------------
    def puct_score(self, child: "Node", c_puct: float) -> float:
        """Compute the PUCT score for `child` from perspective of this node.

        NOTE: this is a PUCT implementation (uses child prior `P`). We name the
        method `puct_score` to avoid confusion with classic UCT. For backward
        compatibility the class also provides `uct_score` as an alias that points
        to the same function.

        Formula used (PUCT):
            score = Q_child + c_puct * P_child * sqrt(max(1, N_parent)) / (1 + N_child)

        Parameters:
        - `child`: child Node whose score is being computed.
        - `c_puct`: exploration constant.
        """
        return puct_score(self, child, c_puct)

    def select_best_puct(self, c_puct: float = 1.0) -> Tuple[Move, "Node"]:
        """Select the child with highest PUCT score.

        Returns (move, child_node). If there are no children, raises ValueError.
        """
        if not self.children:
            raise ValueError("select_best_uct called on a leaf node")
        best_move = None
        best_child = None
        best_score = -float("inf")
        # deterministic iteration order: sort by move to avoid non-determinism
        # micro-optimized PUCT: compute sqrt(parent.N) once and avoid
        # function calls inside the loop for speed.
        sqrt_parent = math.sqrt(max(1, self.N))
        for m in sorted(self.children.keys()):
            ch = self.children[m]
            q = ch.Q if ch.N > 0 else 0.0
            u = c_puct * ch.P * sqrt_parent / (1.0 + ch.N)
            s = q + u
            if s > best_score:
                best_score = s
                best_move = m
                best_child = ch
        assert best_child is not None
        return best_move, best_child

    # -------------------------- backpropagation -------------------------
    def backup(self, value: float):
        """Backpropagate `value` from this node up to the root.

        Value convention: value is from the viewpoint of the root player when the
        backup is initiated. On each step up the tree the value is negated because
        players alternate.
        """
        node = self
        v = float(value)
        while node is not None:
            node.N += 1
            node.W += v
            node.Q = node.W / node.N
            v = -v
            node = node.parent

    # -------------------------- DFPN helpers -----------------------------
    def set_pn_dn(self, pn: float, dn: float):
        """Set proof/disproof numbers for this node.

        DFPN drivers maintain and update these values; the Node just provides
        storage and a convenience setter.
        """
        self.pn = float(pn)
        self.dn = float(dn)

    def mark_proven(self, result: int):
        """Mark this node as proven.

        result: +1 indicates proven win for side to_move, -1 proven loss.
        Search drivers should also store this in a transposition table if used.
        """
        self.proven = int(result)

    # ------------------------- PV / diagnostics -------------------------
    def get_pv(self, max_depth: int = 200) -> List[Move]:
        """Extract a principal variation (sequence of moves) by following best child.

        Heuristic child selection: prefer proven children, then highest Q, then highest P.
        This is a best-effort utility — exact PV extraction for DFPN/MCTS can be
        search-specific and may use TT information.
        """
        pv: List[Move] = []
        node = self
        depth = 0
        while node and node.children and depth < max_depth:
            # prefer proven child (win for side to move)
            best = None
            for ch in node.children.values():
                if ch.proven == 1:
                    best = ch
                    break
            if best is None:
                # otherwise pick child by (Q, P) tie-break
                best = max(node.children.values(), key=lambda c: (c.Q, c.P))
            if best.move is None:
                break
            pv.append(best.move)
            node = best
            depth += 1
        return pv

    # ----------------------- maintenance utilities ----------------------
    def prune_children(self, keep_top_k: int, key: str = "P") -> None:
        """Prune children to keep only top-K by `key` ("P" or "Q" or "N").

        This is handy for VCT/VCF to limit branching and keep the search focused.
        """
        if keep_top_k <= 0:
            self.children.clear()
            return
        if len(self.children) <= keep_top_k:
            return
        if key == "P":
            sorter = lambda kv: kv[1].P
        elif key == "Q":
            sorter = lambda kv: kv[1].Q
        elif key == "N":
            sorter = lambda kv: kv[1].N
        else:
            raise ValueError("unknown key for prune_children")
        # keep top-K moves by sorter (descending)
        items = sorted(self.children.items(), key=sorter, reverse=True)[:keep_top_k]
        self.children = dict(items)

    # Backward-compatibility: some callers may expect `uct_score` as a method name.
    # Provide it as an alias pointing to the correct PUCT implementation.
    uct_score = puct_score


# ------------------ module-level helpers for common tasks ------------------
def select_leaf_by_puct(root: Node, c_puct: float = 1.0) -> Node:
    """Traverse from `root` selecting children by PUCT until a leaf is reached.

    Returns the leaf Node (may be the root itself if it is a leaf).
    """
    node = root
    while not node.is_leaf():
        _, node = node.select_best_uct(c_puct)
    return node


# ------------------ PUCT helper ------------------
def puct_score(parent: Node, child: Node, c_puct: float) -> float:
    """Compute the PUCT score for `child` under `parent`.

    This is the same formula used inside `Node.uct_score`, factored out for
    clarity and to make it easy to unit-test or reuse in other modules.

    Formula:
        score = Q_child + c_puct * P_child * sqrt(max(1, N_parent)) / (1 + N_child)

    - `parent` and `child` are `Node` instances.
    - `c_puct` is the exploration constant (float).
    """
    q = child.Q if child.N > 0 else 0.0
    u = c_puct * child.P * math.sqrt(max(1, parent.N)) / (1.0 + child.N)
    return q + u


# ------------------ Generic alpha-beta (negamax) helper ------------------
def negamax_alpha_beta(position,
                       generate_moves,
                       make_move,
                       evaluate,
                       depth: int,
                       alpha: float = -math.inf,
                       beta: float = math.inf) -> Tuple[float, Optional[Move]]:
    """A small, generic negamax alpha-beta helper.

    This helper is intentionally functional: `make_move` should return a new
    position object (or a lightweight copy) rather than mutating shared state.

    Parameters:
    - position: opaque position object understood by the callbacks.
    - generate_moves(position) -> Iterable[Move]
    - make_move(position, move) -> new_position
    - evaluate(position) -> float   (value from side-to-move's perspective)
    - depth: search depth (int). When depth == 0, `evaluate` is called.
    - alpha, beta: initial window values.

    Returns (best_value, best_move). If there are no legal moves, the function
    calls `evaluate` and returns that value with best_move=None.

    Note: This helper is purposely simple and meant to be called at MCTS leaves
    where a deeper deterministic evaluation (e.g. alpha-beta with quiescence)
    is desired. For large or performance-sensitive searches, replace with a
    domain-optimized implementation that uses incremental make/unmake.
    """
    if depth <= 0:
        return float(evaluate(position)), None

    best_value = -math.inf
    best_move: Optional[Move] = None

    moves = list(generate_moves(position))
    if not moves:
        # no legal moves: evaluate terminal or draw
        return float(evaluate(position)), None

    for mv in moves:
        child_pos = make_move(position, mv)
        val, _ = negamax_alpha_beta(child_pos, generate_moves, make_move, evaluate, depth - 1,
                                     -beta, -alpha)
        val = -val
        if val > best_value:
            best_value = val
            best_move = mv
        alpha = max(alpha, val)
        if alpha >= beta:
            # beta cutoff
            break

    return best_value, best_move


def best_move_by_visits(root: Node) -> Move:
    """Return the child move of `root` with highest visit count N.

    Typically used to pick the final move after MCTS.
    """
    if not root.children:
        raise ValueError("no children to choose from")
    return max(root.children.items(), key=lambda kv: kv[1].N)[0]


def count_nodes(root: Node) -> int:
    """Count nodes in subtree rooted at `root` (simple DFS)."""
    cnt = 0
    stack = [root]
    while stack:
        n = stack.pop()
        cnt += 1
        for ch in n.children.values():
            stack.append(ch)
    return cnt


def node_stats(root: Node) -> Dict[str, int]:
    """Return simple statistics for a subtree: total nodes, proven, visited nodes."""
    total = 0
    proven = 0
    visited = 0
    stack = [root]
    while stack:
        n = stack.pop()
        total += 1
        if n.proven is not None:
            proven += 1
        if n.N > 0:
            visited += 1
        for ch in n.children.values():
            stack.append(ch)
    return {"total": total, "proven": proven, "visited": visited}


__all__ = [
    "Node",
    "select_leaf_by_puct",
    "puct_score",
    "best_move_by_visits",
    "count_nodes",
    "node_stats",
]
