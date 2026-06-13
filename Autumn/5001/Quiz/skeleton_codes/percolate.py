import numpy as np
from mpi4py import MPI
from check_percolate import check_percolate # make sure you have correctly compiled check_percolate.pyx with the given setup.py

def gen_grid(n, p):
    '''
    Generate an n-by-n binary grid with P(1) = p and P(0) = 1-p.

    Args:
        n: Size of the grid
        p: Probability the value of a lattice point is 1 (i.e. "Open")

    Returns:
        A two-dimensional binary numpy array with dtype=np.int32
    '''
    # WRITE YOUR CODE HERE FOR PART (a)
    return np.random.choice([0, 1], size=(n, n), p=[1-p, p]).astype(np.int32)

def percolation_probability(n, p, N=10000):
    '''
    Estimate the probability of percolation for given n and p by running N simulations in parallel using MPI.
    The probability is given by K/N, where K is the number of simulations with percolating grids.

    Args:
        n: Size of the grid
        p: Probability the value of a lattice point is 1 (i.e. "Open")

    Returns:
        A floating-point number
    '''

    # WRITE YOU MPI CODE HERE FOR PART (c)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_N = N // size
    remainder = N % size

    if rank < remainder:
        local_N += 1

    local_count = 0

    np.random.seed(rank + 1)

    for _ in range(local_N):
        grid = gen_grid(n, p).astype(np.int32)
        if check_percolate(grid):
            local_count += 1

    total_count = comm.reduce(local_count, op=MPI.SUM, root=0)

    if rank == 0:
        return float(total_count) / N
    else:
        return None

# DO NOT MODIFY CODES BELOW THIS LINE

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    prob = percolation_probability(50, 0.6)

    if rank == 0:
        print(f"The percolation probability for n=50 and p=0.6 is {prob}")