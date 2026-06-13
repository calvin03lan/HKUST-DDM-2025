import numpy as np

cpdef int check_percolate(int[:, ::1] grid):
    '''
    Args:
        grid: A two-dimensional binary array (with dtype=np.int32)

    Returns:
        1 if the grid percolates, 0 otherwise
    '''
    # FILL IN YOUR CODES HERE FOR PART (b)
    cdef int n = grid.shape[0]
    cdef int i, j, k, l
    cdef int changed
    cdef int[:, ::1] reachable = np.zeros((n, n), dtype=np.int32)

    for j in range(n):
        if grid[0, j] == 1:
            reachable[0, j] = 1

    changed = 1
    while changed:
        changed = 0
        for i in range(n):
            for j in range(n):
                if grid[i, j] == 1 and reachable[i, j] == 0:
                    # Up
                    if i > 0 and reachable[i-1, j] == 1:
                        reachable[i, j] = 1
                        changed = 1
                    # Down
                    elif i < n-1 and reachable[i+1, j] == 1:
                        reachable[i, j] = 1
                        changed = 1
                    # Left
                    elif j > 0 and reachable[i, j-1] == 1:
                        reachable[i, j] = 1
                        changed = 1
                    # Right
                    elif j < n-1 and reachable[i, j+1] == 1:
                        reachable[i, j] = 1
                        changed = 1

    for j in range(n):
        if reachable[n-1, j] == 1:
            return 1

    return 0