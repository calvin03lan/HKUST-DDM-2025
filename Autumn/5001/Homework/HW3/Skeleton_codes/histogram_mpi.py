from mpi4py import MPI
import numpy as np

def parallel_histogram(data, low, high, n_bins):
    """
    Compute histogram using MPI processes.
    
    Args:
        data: an array of numbers (only on rank 0)
        low: the lower range of the bins
        high: the upper range of the bins
        n_bins: number of bins used
    
    Returns:
        An array that contains counts of elements in each bin (only on rank 0)
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # MY ANSWER IS WRITTEN HERE
    if rank == 0:
        data = np.asarray(data, dtype=np.float64)
        N = len(data)
        base,rem = divmod(N,size)

        if base == 0:
            local_data = []
            for i in range(1, min(N, size)):
                comm.send(data[i:i+1], dest=i, tag=0)
            if N > 0:
                local_data = data[0:1]
            else:
                local_data = np.array([])
            for i in range(N, size):
                comm.send(np.array([]), dest=i, tag=0)
        else:
            for i in range(1, size):
                start = i * base
                end = start + base
                comm.send(data[start:end], dest=i, tag=0)
            local_data = data[0:base + rem]
    else:
        local_data = comm.recv(source=0, tag=0)
        local_data = np.asarray(local_data, dtype=np.float64)

    bin_edges = np.linspace(low, high, n_bins + 1)
    local_hist, _ = np.histogram(local_data, bins=bin_edges)

    if rank == 0:
        global_hist = np.empty(n_bins, dtype=local_hist.dtype)
    else:
        global_hist = None

    comm.Reduce(local_hist, global_hist, op=MPI.SUM, root=0)

    return global_hist

# DO NOT MODIFY ANYTHING BELOW THIS POINT IN YOUR SUBMITTED CODE
def main():
    """
    Compute histogram using MPI
    Run with: mpiexec -n <num_processes> python histogram_mpi.py
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        test_data = np.random.rand(100000)
        low = 0.0
        high = 1.0
        n_bins = 50

        hist = parallel_histogram(test_data, low, high, n_bins)
        print(f"Histogram: {hist}")

if __name__ == "__main__":
    main()