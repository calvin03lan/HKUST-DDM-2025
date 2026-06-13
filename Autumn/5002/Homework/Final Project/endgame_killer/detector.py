''' Module for all helper functions related to pattern detection. 

Fundamental functions: 
        Given fundamental functions for position and board analysis.

Pattern definitions:
        Define threat patterns as dicts of lists.

Threat detection functions: 
        Functions to detect various threat types on the board.
 '''


#==============================================================================================================
# Fundamental functions
#==============================================================================================================

def _board_to_lines(board):
    '''Convert the board state to 3 lists of lines: vertical, horizontal, and diagonal.
    Horizontal lines run from left to right.
    vert_lines: list of vertical lines
    Diagonal lines run from top-left to bottom-right.

    returns:
        hori_lines: list of horizontal lines (with boundaries)
        vert_lines: list of vertical lines (with boundaries)
        diag_lines: list of diagonal lines (with boundaries)

    Note that each line has a boundary value (5) at both ends, the boundaries are essential
    for correct pattern detection.
    e.g. [5, -1, -1, ..., 1, 0, 5]
    ''' 
    hori_lines = []
    vert_lines = []
    diag_lines = []
    
    # Horizontal lines
    for u in range(15):
        line = [5]  # boundary
        for v in range(15 - u):
            line.append(board[u, v])
        line.append(5)  # boundary
        hori_lines.append(line)     

    # Vertical lines
    for v in range(15):
        line = [5]  # boundary
        for u in range(15 - v):
            line.append(board[u, v])
        line.append(5)  # boundary
        vert_lines.append(line)

    # Diagonal lines
    for u in range(15):
        line = [5] # boundary
        # include v == u so the top-most cell (row 0, col u) is part of the diagonal
        for v in range(u + 1):
            line.append(board[u - v, v])
        line.append(5)  # boundary
        diag_lines.append(line)
    
    return hori_lines, vert_lines, diag_lines


def _is_valid(board, u, v):
    """Return True if (u, v) is a valid playable and empty cell on the triangular board."""
    n = board.shape[0]
    return 0 <= u < n and 0 <= v < n - u and board[u, v] == 0


def has_nhd(board, u, v, color=None):
    """Return True if (u, v) has any non-empty neighbour on the three legal axes.

    Semantics:
    - Checks the six adjacent offsets along the three principal axes of the triangular
      board: horizontal, vertical and the up-right diagonal (both directions).
    - If `color` is None, returns True if any neighbour is occupied by any stone (1 or -1).
    - If `color` is provided (1 or -1), returns True only if a neighbour of that color exists.

    The function performs bounds checks and ignores sentinel/boundary cells.
    """
    n = board.shape[0]
    # validate central coordinate is on-board (we don't require it to be empty)
    if not (0 <= u < n and 0 <= v < n - u):
        return False

    # neighbour offsets: both directions for the three principal axes
    neigh_offsets = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, -1), (-1, 1)]
    for du, dv in neigh_offsets:
        x, y = u + du, v + dv
        if 0 <= x < n and 0 <= y < n - x:
            val = board[x, y]
            # ignore boundary sentinel values (5) and empty (0)
            if val == 0 or val == 5:
                continue
            if color is None:
                return True
            if val == color:
                return True
    return False


# detect if placing a stone creates an overline (>5 in a row)
def _creates_overline(board, u, v, color):
    """Return True if placing `color` at (u,v) would create a contiguous run longer than 5.
    This simulates the placement and scans the three principal axes.
    """
    n = board.shape[0]
    # quick sanity: must be on-board and currently empty
    if not (0 <= u < n and 0 <= v < n - u):
        return False
    if board[u, v] != 0:
        return False

    bcopy = board.copy()
    bcopy[u, v] = color

    dirs = [(0, 1), (1, 0), (-1, 1)]  # horizontal, vertical, diag
    for du, dv in dirs:
        count = 1
        # forward
        x, y = u + du, v + dv
        while 0 <= x < n and 0 <= y < n - x and bcopy[x, y] == color:
            count += 1
            x += du; y += dv
        # backward
        x, y = u - du, v - dv
        while 0 <= x < n and 0 <= y < n - x and bcopy[x, y] == color:
            count += 1
            x -= du; y -= dv
        if count > 5:
            return True
    return False



#==============================================================================================================
# Threat pattern definitions    
#==============================================================================================================

def type4_threats(color):
    ''' Return dictionary of sets of each kind of type-4 threat patterns created by the given color.'''
    X = color # own color
    O = -color # opponent's color
    B = 5 # boundary
    # use lists-of-lists for patterns (easier to compare with slices)
    type4_threats = {
        'open4': [[0, X, X, X, X, 0]],
        'closed4': [[O, X, X, X, X, 0], [0, X, X, X, X, O],
                    [B, X, X, X, X, 0], [0, X, X, X, X, B]],
        'jump4': [[X, 0, X, X, X], [X, X, 0, X, X], [X, X, X, 0, X]]
    }
    return type4_threats


def type3_lethal_threats(color):
    ''' Return dictionary of sets of each kind of type-3 lethal threat patterns created by the given color.'''
    X = color # own color
    O = -color # opponent's color
    B = 5 # boundary
    type3_threats = {
        'open3': [[0, 0, X, X, X, 0], [0, X, X, X, 0, 0]],
        'jump3': [[0, X, 0, X, X, 0], [0, X, X, 0, X, 0]]
    }
    return type3_threats


def type3_nonlethal_threats(color):
    ''' Return dictionary of sets of each kind of type-3 non-lethal threat patterns created by the given color.'''
    X = color # own color
    O = -color # opponent's color
    B = 5 # boundary
    type3_threats = {
        'closed3': [[O, X, X, X, 0, 0], [0, 0, X, X, X, O],
                    [B, X, X, X, 0, 0], [0, 0, X, X, X, B]],
        'blocked-jump3': [[0, X, 0, X, X, O], [O, X, X, 0, X, 0],
                        [0, X, 0, X, X, B], [B, X, X, 0, X, 0]],
        'blocked-open3': [[O, 0, X, X, X, 0, O], [B, 0, X, X, X, 0, B],
                          [O, 0, X, X, X, 0, B], [B, 0, X, X, X, 0, O]]
    }
    return type3_threats


def type2_threats(color):
    ''' Return dictionary of sets of each kind of type-2 threat patterns created by the given color.'''
    X = color # own color
    O = -color # opponent's color
    B = 5 # boundary
    type2_threats = {
        'open2': [[0, X, X, 0]],
        'jump2': [[0, X, 0, X, 0]]
    }
    return type2_threats


def weak_type2_threats(color):
    ''' Return dictionary of sets of each kind of weak type-2 threat patterns created by the given color.'''
    X = color # own color
    O = -color # opponent's color
    B = 5 # boundary
    weak_type2_threats = {
        'closed2': [[O, X, X, 0, 0], [0, 0, X, X, O], 
                    [B, X, X, 0, 0], [0, 0, X, X, B]]
    }
    return weak_type2_threats


#==============================================================================================================
# Threat detection functions
#==============================================================================================================

def pattern_to_moves(line, pattern):
    """ Find the pattern in the given line and return the set of coordinates of 0's in the 
    pattern (i.e. the position we want to attack/block).
    """
    moves = set()
    n, m = len(line), len(pattern)
    if m == 0 or n < m:
        return moves

    # The boundaries are excluded from the line for faster 0 matching (since boundaries are never 0).
    inner = line[1:-1]

    # If the pattern does NOT contain the boundary token (5), match against inner cells
    # and return indices relative to the inner list (no -1 adjustment needed).
    if 5 not in pattern:
        ni = len(inner)
        if m > ni:
            return moves
        for i in range(ni - m + 1):
            window = inner[i:i + m]
            if (isinstance(pattern, tuple) and tuple(window) == pattern) or (isinstance(pattern, list) and window == pattern):
                for k, pval in enumerate(pattern):
                    if pval == 0:
                        moves.add(i + k)
        return moves

    # Otherwise (pattern includes boundary token), fall back to matching the full line
    # and map matched '0' positions to inner-cell indices by subtracting 1 from full-line index.
    for i in range(n - m + 1):
        window = line[i:i + m]
        if (isinstance(pattern, tuple) and tuple(window) == pattern) or (isinstance(pattern, list) and window == pattern):
            for k, pval in enumerate(pattern):
                if pval == 0:
                    idx_in_line = i + k
                    if 1 <= idx_in_line <= n - 2:
                        cell_pos = idx_in_line - 1
                        moves.add(cell_pos)
    return moves


def lethal_type4_moves(board, color):
    ''' This function will detect all type-4 threats (e.g. open4, closed4) created by player of `color` and
    return a set of moves for immediately win for `color`.
    Args:
        board: 15x15 numpy array representing the board state
        color: int, the color of AI
    Returns:
        Set of (u, v) tuples representing winning moves
    '''
    moves = set()

    hori_lines, vert_lines, diag_lines = _board_to_lines(board)
    type4_patterns = type4_threats(color)

    # Horizontal lines: line index -> row u, cell index -> column v
    for u, line in enumerate(hori_lines):
        for pats in type4_patterns.values():
            for pat in pats:
                idxs = pattern_to_moves(line, pat)
                for cell_idx in idxs:
                    v = cell_idx
                    # validate coordinate (v must fit row length in triangular board)
                    if _is_valid(board, u, v):
                        moves.add((u, v))

    # Vertical lines: line index -> column v, cell index -> row u
    for v, line in enumerate(vert_lines):
        for pats in type4_patterns.values():
            for pat in pats:
                idxs = pattern_to_moves(line, pat)
                for cell_idx in idxs:
                    u = cell_idx
                    if _is_valid(board, u, v):
                        moves.add((u, v))

    # Diagonal lines: line index li corresponds to parameter U in _board_to_lines;
    # cell index j maps to (row = li - j, col = j)
    for li, line in enumerate(diag_lines):
        for pats in type4_patterns.values():
            for pat in pats:
                idxs = pattern_to_moves(line, pat)
                for j in idxs:
                    u = li - j
                    v = j
                    if _is_valid(board, u, v):
                        moves.add((u, v))
    # filter out moves that would create an overline (>5 contiguous stones)
    filtered = set()
    for (uu, vv) in moves:
        if not _creates_overline(board, uu, vv, color):
            filtered.add((uu, vv))
    return filtered


def lethal_type3_moves(board, color):
    ''' This function will detect all type-3 threats created by player of `color` and
    return a set of moves that create open4 threats (i.e. the '0' positions for type-3 threats) for `color`.
    Args:
        board: 15x15 numpy array representing the board state
        color: int, the color of the AI
    Returns:
        Set of (u, v) tuples representing moves that create open4 threats
    '''
    moves = set()

    hori_lines, vert_lines, diag_lines = _board_to_lines(board)
    type3_patterns = type3_lethal_threats(color)

    # Horizontal: line index -> row u, cell index -> column v
    for u, line in enumerate(hori_lines):
        for pats in type3_patterns.values():
            for pat in pats:
                cell_idxs = pattern_to_moves(line, pat)
                for v in cell_idxs:
                    if _is_valid(board, u, v):
                        moves.add((u, v))

    # Vertical: line index -> column v, cell index -> row u
    for v, line in enumerate(vert_lines):
        for pats in type3_patterns.values():
            for pat in pats:
                cell_idxs = pattern_to_moves(line, pat)
                for u in cell_idxs:
                    if _is_valid(board, u, v):
                        moves.add((u, v))

    # Diagonal: diag index li corresponds to parameter U in _board_to_lines;
    # cell index j maps to (u = li - j, v = j)
    for li, line in enumerate(diag_lines):
        for pats in type3_patterns.values():
            for pat in pats:
                cell_idxs = pattern_to_moves(line, pat)
                for j in cell_idxs:
                    u = li - j
                    v = j
                    if _is_valid(board, u, v):
                        moves.add((u, v))
    # filter fake threats that produce >5 in a row
    filtered = set()
    for (uu, vv) in moves:
        if not _creates_overline(board, uu, vv, color):
            filtered.add((uu, vv))
    return filtered


def nonlethal_type3_moves(board, color):
    ''' This function will detect all type-3 non-lethal threats created by player of `color` and
    return a set of moves that create/block type-3 non-lethal threats for `color`.
    Args:
        board: 15x15 numpy array representing the board state
        color: int, the color of the AI
    Returns:
        Set of (u, v) tuples representing moves that create/block type-3 non-lethal threats
    '''
    moves = set()

    hori_lines, vert_lines, diag_lines = _board_to_lines(board)
    patterns = type3_nonlethal_threats(color)

    # Horizontal
    for u, line in enumerate(hori_lines):
        for pats in patterns.values():
            for pat in pats:
                idxs = pattern_to_moves(line, pat)
                for cell_idx in idxs:
                    v = cell_idx
                    if _is_valid(board, u, v):
                        moves.add((u, v))

    # Vertical
    for v, line in enumerate(vert_lines):
        for pats in patterns.values():
            for pat in pats:
                idxs = pattern_to_moves(line, pat)
                for cell_idx in idxs:
                    u = cell_idx
                    if _is_valid(board, u, v):
                        moves.add((u, v))

    # Diagonal: li -> param U in _board_to_lines; cell j -> (row = li - j, col = j)
    for li, line in enumerate(diag_lines):
        for pats in patterns.values():
            for pat in pats:
                idxs = pattern_to_moves(line, pat)
                for j in idxs:
                    row = li - j
                    col = j
                    if _is_valid(board, row, col):
                        moves.add((row, col))
    # remove moves that would create overlines (>5)
    filtered = set()
    for (uu, vv) in moves:
        if not _creates_overline(board, uu, vv, color):
            filtered.add((uu, vv))
    return filtered


def type2_moves(board, color):
    """ Detect type-2 threats for `color` and return set of moves that create/block them.

    Args:
        board: 15x15 numpy array representing the board state
        color: int, the color of the AI
    Returns:
        Set of (u, v) tuples representing moves that create/block type-2 threats
    """
    moves = set()

    hori_lines, vert_lines, diag_lines = _board_to_lines(board)
    patterns = type2_threats(color)

    # Horizontal
    for u, line in enumerate(hori_lines):
        for pats in patterns.values():
            for pat in pats:
                idxs = pattern_to_moves(line, pat)
                for cell_idx in idxs:
                    v = cell_idx
                    if _is_valid(board, u, v):
                        moves.add((u, v))

    # Vertical
    for v, line in enumerate(vert_lines):
        for pats in patterns.values():
            for pat in pats:
                idxs = pattern_to_moves(line, pat)
                for cell_idx in idxs:
                    u = cell_idx
                    if _is_valid(board, u, v):
                        moves.add((u, v))

    # Diagonal
    for li, line in enumerate(diag_lines):
        for pats in patterns.values():
            for pat in pats:
                idxs = pattern_to_moves(line, pat)
                for j in idxs:
                    row = li - j
                    col = j
                    if _is_valid(board, row, col):
                        moves.add((row, col))
    # filter out any move producing more-than-5 contiguous stones
    filtered = set()
    for (uu, vv) in moves:
        if not _creates_overline(board, uu, vv, color):
            filtered.add((uu, vv))
    return filtered


def weak_type2_moves(board, color):
    """Detect weak type-2 threats for `color` and return set of moves that create/block them.

    This corresponds to patterns returned by `weak_type2_threats(color)`.
    """
    moves = set()

    hori_lines, vert_lines, diag_lines = _board_to_lines(board)
    patterns = weak_type2_threats(color)

    # Horizontal
    for u, line in enumerate(hori_lines):
        for pats in patterns.values():
            for pat in pats:
                idxs = pattern_to_moves(line, pat)
                for cell_idx in idxs:
                    v = cell_idx
                    if _is_valid(board, u, v):
                        moves.add((u, v))

    # Vertical
    for v, line in enumerate(vert_lines):
        for pats in patterns.values():
            for pat in pats:
                idxs = pattern_to_moves(line, pat)
                for cell_idx in idxs:
                    u = cell_idx
                    if _is_valid(board, u, v):
                        moves.add((u, v))

    # Diagonal
    for li, line in enumerate(diag_lines):
        for pats in patterns.values():
            for pat in pats:
                idxs = pattern_to_moves(line, pat)
                for j in idxs:
                    row = li - j
                    col = j
                    if _is_valid(board, row, col):
                        moves.add((row, col))
    # exclude weak-type2 moves that actually create overlines
    filtered = set()
    for (uu, vv) in moves:
        if not _creates_overline(board, uu, vv, color):
            filtered.add((uu, vv))
    return filtered


    # Double-threat helper functions were removed during refactor.
    # Tactical double-threat detection is now handled by the VCT proof search
    # and `if_exist_threats` which uses lightweight detectors to decide
    # whether to invoke the proof search. The explicit `is_double_threat_move`
    # and `double_threat_moves` helpers are intentionally omitted.


#==============================================================================================================
# Other Dectector Functions
#==============================================================================================================

def candidates_moves(board, color):
    """Return (candidates_set, priority_level).

        priority_level: lower means higher priority:
            1 = own immediate win (type4)
            2 = opponent immediate win (must block)
            3 = own lethal type3
            4 = opponent lethal type3
            99 = fallback (many candidates)
    """
    own = color  # own color
    opp = -color  # opponent color
    candidates = set()

    # 1) immediate own wins (type4)
    type4_moves_own = lethal_type4_moves(board, own)
    if type4_moves_own:
        return set(type4_moves_own), 1

    # 2) immediate opponent wins (need to block)
    type4_moves_opp = lethal_type4_moves(board, opp)
    if type4_moves_opp:
        return set(type4_moves_opp), 2

    # 3) lethal type3
    type3_moves_own = lethal_type3_moves(board, own)
    if type3_moves_own:
        return set(type3_moves_own), 3

    type3_moves_opp = lethal_type3_moves(board, opp)
    if type3_moves_opp:
        return set(type3_moves_opp), 4

    # 4) fallback: all available moves
    for u in range(board.shape[0]):
        for v in range(board.shape[0] - u):
            if _is_valid(board, u, v):
                candidates.add((u, v))
    return candidates, 99


def if_exist_threats(board, color):
    """Return True if there exists any tactical threat (type2, weak-type2,
    type3 or type4) for either side near the area of play.

    The function calls the lightweight detectors and returns a boolean.
    This is intended to be used to decide whether to invoke an expensive
    proof search (VCT) even when `candidates_moves` returns the fallback set.
    """
    own = color
    opp = -color

    # immediate wins
    if lethal_type4_moves(board, own) or lethal_type4_moves(board, opp):
        return True

    # lethal type-3
    if lethal_type3_moves(board, own) or lethal_type3_moves(board, opp):
        return True

    # type-2 and weak type-2 (can combine into higher threats)
    if type2_moves(board, own) or type2_moves(board, opp):
        return True
    if weak_type2_moves(board, own) or weak_type2_moves(board, opp):
        return True

    return False


def check_winner(board):
    """Detect a winner on the board.

    Returns (winner, info) where winner is:
      1  -> black wins
     -1  -> white wins
      2  -> draw (no empties)
      0  -> game ongoing

    `info` is currently None (kept for compatibility with utility_func.check_winner signature).
    """
    hori_lines, vert_lines, diag_lines = _board_to_lines(board)

    def scan_lines(lines):
        for line in lines:
            # strip boundaries
            inner = line[1:-1]
            if not inner:
                continue
            # scan for runs
            current = None
            run = 0
            for val in inner:
                if val == 0 or val == 5:
                    current = None
                    run = 0
                    continue
                if val == current:
                    run += 1
                else:
                    current = val
                    run = 1
                if run >= 5:
                    return current
        return None

    for lines in (hori_lines, vert_lines, diag_lines):
        res = scan_lines(lines)
        if res is not None:
            return int(res), None

    # no winner: check for draw (no empties)
    if not any(_is_valid(board, u, v) for u in range(board.shape[0]) for v in range(board.shape[0]-u)):
        return 2, None

    return 0, None