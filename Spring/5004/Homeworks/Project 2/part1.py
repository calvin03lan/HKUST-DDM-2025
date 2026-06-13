import numpy as np
import matplotlib.pyplot as plt

def sample_standard_normal(num_samples):
    """
    Sample from standard normal distribution N(0, 1) using rejection sampling.
    Target: p(w) = 1/sqrt(2*pi) * exp(-w^2/2)
    Proposal: q(w) = Laplace(0, 1) = 1/2 * exp(-|w|)
    Comparison function: f(w) = A * q(w) = sqrt(e/(2*pi)) * exp(-|w|)
    """
    samples = []
    total_trials = 0
    
    # Pre-compute A to avoid recalculation
    A = np.sqrt(np.e / (2 * np.pi))
    
    while len(samples) < num_samples:
        # Sample w from Laplace distribution (lambda = 1)
        # Note: Laplace(0, 1) can be generated using double exponential
        u1 = np.random.uniform(-0.5, 0.5)
        w = -np.sign(u1) * np.log(1 - 2 * np.abs(u1))
        
        # Sample u uniformly from [0, f(w)]
        f_w = A * np.exp(-np.abs(w))
        u = np.random.uniform(0, f_w)
        
        # Target probability p(w)
        p_w = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * w**2)
        
        # Accept if u <= p(w)
        if u <= p_w:
            samples.append(w)
        total_trials += 1
        
    acceptance_rate = num_samples / total_trials
    return np.array(samples), acceptance_rate

def sample_normal(num_samples, mu, sigma):
    """
    Sample from general normal distribution N(mu, sigma^2) using rejection sampling.
    First sample Z ~ N(0, 1), then transform X = mu + sigma * Z
    """
    z_samples, acceptance_rate = sample_standard_normal(num_samples)
    x_samples = mu + sigma * z_samples
    return x_samples, acceptance_rate

if __name__ == "__main__":
    np.random.seed(42)
    num_samples_to_generate = 100000
    
    # ---------------------------------------------------------
    # (d) & (e) Sample N(0, 1) and plot
    # ---------------------------------------------------------
    print(f"Generating {num_samples_to_generate} samples for N(0, 1)...")
    standard_samples, acc_rate = sample_standard_normal(num_samples_to_generate)
    
    theoretical_acc_rate = np.sqrt(np.pi / (2 * np.e))
    print(f"Actual Acceptance Rate:     {acc_rate:.4f}")
    print(f"Theoretical Acceptance Rate: {theoretical_acc_rate:.4f}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot histogram of samples
    count, bins, ignored = plt.hist(standard_samples, bins=100, density=True, 
                                    alpha=0.6, color='b', label='Sampled Data')
    
    # Plot theoretical N(0,1) curve
    w_vals = np.linspace(-4, 4, 1000)
    p_vals = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * w_vals**2)
    plt.plot(w_vals, p_vals, 'r', linewidth=2, label=r'Theoretical $\mathcal{N}(0, 1)$')
    
    plt.title('Rejection Sampling for Standard Normal Distribution')
    plt.xlabel('w')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig('part1_standard_normal.png', dpi=300, bbox_inches='tight')
    print("Saved plot to part1_standard_normal.png")
    
    # ---------------------------------------------------------
    # (f) Sample general normal N(mu, sigma^2)
    # ---------------------------------------------------------
    mu_test = 5.0
    sigma_test = 2.0
    print(f"\nGenerating {num_samples_to_generate} samples for N({mu_test}, {sigma_test}^2)...")
    general_samples, _ = sample_normal(num_samples_to_generate, mu_test, sigma_test)
    
    print(f"Sample Mean: {np.mean(general_samples):.4f} (Expected: {mu_test})")
    print(f"Sample Variance: {np.var(general_samples):.4f} (Expected: {sigma_test**2})")
    
