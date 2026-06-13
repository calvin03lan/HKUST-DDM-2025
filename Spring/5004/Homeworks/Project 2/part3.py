import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ---------------------------------------------------------
# Helper Functions (provided in PDF)
# ---------------------------------------------------------
def log_likelihood(w, X, y):
    z = y * (X @ w)
    log_sigma = -np.log1p(np.exp(-z))
    return np.sum(log_sigma)

def log_prior(w, alpha=1.0):
    d = len(w)
    return -0.5 * alpha * np.dot(w, w) - 0.5 * d * np.log(2*np.pi/alpha)

def log_posterior(w, X, y, alpha=1.0):
    return log_likelihood(w, X, y) + log_prior(w, alpha)

def unnormalized_posterior(w, X, y, alpha=1.0):
    return np.exp(log_posterior(w, X, y, alpha))

# ---------------------------------------------------------
# Data Generation (Same as Part 2)
# ---------------------------------------------------------
def generate_data(N=500):
    np.random.seed(42)
    w_true = np.array([-1, 1])
    x1 = np.random.randn(N) + 1  
    X = np.column_stack((np.ones(N), x1)) 
    epsilon = np.random.normal(0, 2, N) 
    y = np.sign(X @ w_true + epsilon)
    y[y == 0] = 1 
    return X, y

# ---------------------------------------------------------
# MCMC & Optimization Methods
# ---------------------------------------------------------
def metropolis_hastings(X, y, num_samples, sigma_p):
    """Metropolis-Hastings algorithm for sampling from posterior."""
    w_current = np.array([0.0, 0.0]) # Initial guess
    samples = np.zeros((num_samples, 2))
    accepted = 0
    
    log_post_current = log_posterior(w_current, X, y)
    
    for i in range(num_samples):
        # Proposal: w' = w + delta, delta ~ N(0, sigma_p^2 I)
        delta = np.random.normal(0, sigma_p, 2)
        w_proposed = w_current + delta
        
        log_post_proposed = log_posterior(w_proposed, X, y)
        
        # Acceptance ratio in log space: log(r) = log(p') - log(p)
        # Since proposal is symmetric, q(w'|w) = q(w|w'), they cancel out
        log_r = log_post_proposed - log_post_current
        
        if np.log(np.random.uniform()) < log_r:
            w_current = w_proposed
            log_post_current = log_post_proposed
            accepted += 1
            
        samples[i] = w_current
        
    acc_rate = accepted / num_samples
    return samples, acc_rate

def simulated_annealing(X, y, initial_temp=10.0, cooling_rate=0.99, max_iters=5000):
    """
    Simulated Annealing to find MAP.
    Energy function E(w) = -log[p(D|w)p(w)] = -log_posterior
    """
    w_current = np.array([0.0, 0.0])
    T = initial_temp
    
    best_w = w_current
    best_E = -log_posterior(w_current, X, y)
    
    current_E = best_E
    
    for i in range(max_iters):
        # Propose neighbor
        w_proposed = w_current + np.random.normal(0, 0.1, 2)
        proposed_E = -log_posterior(w_proposed, X, y)
        
        delta_E = proposed_E - current_E
        
        # Accept if better (delta_E < 0) or with probability exp(-delta_E / T)
        if delta_E < 0 or np.random.uniform() < np.exp(-delta_E / T):
            w_current = w_proposed
            current_E = proposed_E
            
            if current_E < best_E:
                best_E = current_E
                best_w = w_current
                
        # Cool down
        T *= cooling_rate
        if T < 1e-8:
            break
            
    return best_w

def importance_sampling(X, y, w_map, cov_matrix, num_samples=1000000):
    """
    Importance Sampling for computing marginal likelihood.
    q(w) = N(w_map, cov_matrix)
    """
    # Sample from proposal distribution q(w)
    samples = np.random.multivariate_normal(w_map, cov_matrix, num_samples)
    
    # We need p(D|w)p(w) / q(w) for each sample
    # Log space calculation for numerical stability
    weights = np.zeros(num_samples)
    
    from scipy.stats import multivariate_normal
    q_dist = multivariate_normal(mean=w_map, cov=cov_matrix)
    
    for i in range(num_samples):
        w = samples[i]
        log_p = log_posterior(w, X, y)
        log_q = q_dist.logpdf(w)
        weights[i] = np.exp(log_p - log_q)
        
    Z = np.mean(weights)
    err = np.std(weights) / np.sqrt(num_samples)
    return Z, err

if __name__ == "__main__":
    X, y = generate_data()

    # ---------------------------------------------------------
    # Part 3 (a): Metropolis Algorithm
    # ---------------------------------------------------------
    print("Part 3 (a): Metropolis Algorithm")
    num_samples = 10000
    # Tuned to get an acceptance rate close to the requested 30%.
    sigma_p = 0.17

    samples, acc_rate = metropolis_hastings(X, y, num_samples, sigma_p)
    print(f"  Acceptance Rate: {acc_rate * 100:.2f}% (Target: ~30%)")

    # Discard burn-in (first 1000 samples)
    burn_in = 1000
    valid_samples = samples[burn_in:]

    sample_mean = np.mean(valid_samples, axis=0)
    sample_cov = np.cov(valid_samples.T)
    print(f"  Sample Mean: {sample_mean}")
    print(f"  Sample Covariance:\n{sample_cov}\n")

    # ---------------------------------------------------------
    # Part 3 (b): Simulated Annealing (MAP)
    # ---------------------------------------------------------
    print("Part 3 (b): Simulated Annealing for MAP")
    w_map = simulated_annealing(X, y)
    print(f"  Found MAP estimate: w_MAP = {w_map}")
    print(f"  Log-Posterior at MAP: {log_posterior(w_map, X, y):.4f}\n")

    # ---------------------------------------------------------
    # Part 3 (c): Importance Sampling
    # ---------------------------------------------------------
    print("Part 3 (c): Importance Sampling")
    # Proposal distribution q(w): Gaussian peaked at MAP, width estimated from MCMC
    # Using sample covariance from (a)
    print(f"  Using Proposal q(w) = N(w_MAP, sample_cov)")

    Z_is, err_is = importance_sampling(X, y, w_map, sample_cov, num_samples=1000000)

    print(f"  Marginal Likelihood (IS): {Z_is:.10e}")
    print(f"  Standard Error      (IS): {err_is:.10e}")
    print(f"  Relative Error          : {(err_is/Z_is)*100:.4f}%")
    print(f"\n  Recall Part 2 (d) Simpson's: 1.4473757050e-143")
    print(f"  Recall Part 2 (e) Crude MC : 1.4426724163e-143 (Std Err: 1.28e-145)")
    print(f"  The IS standard error is dramatically smaller, showing variance reduction.\n")

    # ---------------------------------------------------------
    # Plot: MCMC samples vs Theoretical Posterior Contour
    # ---------------------------------------------------------
    p_D = Z_is

    plt.figure(figsize=(10, 8))

    plt.plot(valid_samples[:, 0], valid_samples[:, 1], '.', alpha=0.1, markersize=5, label='MCMC Samples')

    std0 = np.sqrt(sample_cov[0, 0])
    std1 = np.sqrt(sample_cov[1, 1])
    margin = 3.5
    w0_vals = np.linspace(sample_mean[0] - margin * std0, sample_mean[0] + margin * std0, 100)
    w1_vals = np.linspace(sample_mean[1] - margin * std1, sample_mean[1] + margin * std1, 100)
    W0, W1 = np.meshgrid(w0_vals, w1_vals)
    Z = np.zeros_like(W0)

    for i in range(len(w0_vals)):
        for j in range(len(w1_vals)):
            # p(w|D) = p(D|w)p(w) / p(D)
            Z[j, i] = unnormalized_posterior(np.array([W0[j, i], W1[j, i]]), X, y) / p_D

    contour = plt.contour(W0, W1, Z, levels=10, cmap='viridis')
    plt.colorbar(contour, label='Density p(w|D)')

    plt.title('MCMC Samples vs Theoretical Posterior Distribution')
    plt.xlabel('w0 (bias)')
    plt.ylabel('w1 (weight)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('part3_metropolis.png', dpi=300, bbox_inches='tight')
    print("Saved plot to part3_metropolis.png")
