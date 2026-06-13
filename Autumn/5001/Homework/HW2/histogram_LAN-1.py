import threading
import time
import numpy as np

def worker(thread_id, data_chunk, low, high, n_bins, results, lock):
    print(f"[Thread {thread_id}] Processed {len(data_chunk)} items") # Check if thread correctly 
    local_hist, _ = np.histogram(data_chunk, bins=n_bins, range=(low, high)) # done entirely locally, no locks required 
    with lock:
        for i in range(n_bins):
            results[i] += int(local_hist[i]) # lock to ensure only 1 thread can modify the total result at a time

def serial_histogram(data, low, high, n_bins):
    hist, _ = np.histogram(data, bins=n_bins, range=(low, high))
    return hist.tolist()

def parallel_histogram(data, low, high, n_bins, n_threads): # 5 specified parameters
    results = [0] * n_bins
    lock = threading.Lock() # Get thread lock
    chunk_size = len(data) // n_threads
    threads = []

    for i in range(n_threads):
        start = i * chunk_size
        end = start + chunk_size if i != n_threads - 1 else len(data)
        chunk = data[start:end]
        t = threading.Thread(target=worker, args=(i, chunk, low, high, n_bins, results, lock))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    return results

def main():
    test_data = np.random.rand(100000)
    n_threads = 8
    low = 0.0
    high = 1.0
    n_bins = 50

    start = time.time()
    serial_hist = serial_histogram(test_data, low, high, n_bins)
    elapsed_serial = time.time() - start

    start = time.time()
    parallel_hist = parallel_histogram(test_data, low, high, n_bins, n_threads)
    elapsed_parallel = time.time() - start

    # Output
    # print("Serial histogram:", serial_hist)
    # print("Parallel histogram:", parallel_hist)
    print(f"Results match: {serial_hist == parallel_hist}") # Check if results are same
    print(f"Serial time: {elapsed_serial:.4f}s")
    print(f"Parallel time: {elapsed_parallel:.4f}s")
    if elapsed_serial > 0:
        print(f"Speed up: { elapsed_parallel/elapsed_serial:.2f}x")
    else:
        print("Parallel time too small to compute speedup.")

if __name__ == "__main__":
    main()