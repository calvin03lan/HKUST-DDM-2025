# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython

cdef initialize(double phi_top, double phi_bottom, double phi_left, double phi_right, int n):
    '''
    Initialize the grid satisfying the boundary conditions with zero in the interior

    Args:
        phi_top: value of phi at the top edge
        phi_bottom: value of phi at the bottom edge
        phi_left: value of phi at the left edge
        phi_right: value of phi at the right edge
        n: size of grid (minus 1)

    Returns:
        A 2D numpy array
    '''
    cdef double[:, :] phi = np.zeros((n + 1, n + 1), dtype=np.float64)
    phi[0,   :] = phi_top
    phi[-1,  :] = phi_bottom
    phi[:,   0] = phi_left
    phi[:,  -1] = phi_right
    return phi

# TO DO: FILL IN THE APPROPRIATE ARGUMENTS IN update
@cython.boundscheck(False)  # Turn off bounds checking for speed
@cython.wraparound(False)   # Turn off negative indexing for speed
cdef update(double[:, :] phi_new, double[:, :] phi_old, int n):
    '''
    Update phi_new in-place using values of phi_old

    Args:
        phi_new: the updated grid values (memory view)
        phi_old: the previous grid values (memory view)

    Returns:
        None
    '''
    cdef int i, j
    for i in range(1, n):
        for j in range(1, n):
            phi_new[i, j] = 0.25 * (phi_old[i+1, j] + phi_old[i-1, j] +
                                    phi_old[i, j+1] + phi_old[i, j-1])

def solve_cython(boundary_conditions, n=100, iter_max=10000, tol=1e-5):
    '''
    Solving the Laplace equation with Jacobi's iteration (Cython version)

    Args:
        boundary_conditions: a tuple of (phi_top, phi_bottom, phi_left, phi_right)
        n: size of grid (minus 1), default to 100
        iter_max: maximum number of iterations allowed, default to 10000
        tol: tolerance level, default to 1e-5

    Returns:
        A 2D numpy array corresponding to the solution
    '''
    cdef double phi_top, phi_bottom, phi_left, phi_right
    phi_top, phi_bottom, phi_left, phi_right = boundary_conditions

    cdef double[:, :] phi_old = initialize(phi_top, phi_bottom, phi_left, phi_right, n)
    cdef double[:, :] phi_new = initialize(phi_top, phi_bottom, phi_left, phi_right, n)

    cdef double max_diff
    cdef int iter = 0

    while iter <= iter_max:
        update(phi_new, phi_old, n)
        max_diff = np.max(np.abs(np.asarray(phi_new) - np.asarray(phi_old)))
        if max_diff < tol:
            break
        phi_old, phi_new = phi_new, phi_old
        iter += 1

    return np.asarray(phi_new)
