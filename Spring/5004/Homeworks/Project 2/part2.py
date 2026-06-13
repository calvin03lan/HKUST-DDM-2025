import numpy as np

# ---------------------------------------------------------
# Helper Functions (provided in PDF)
# ---------------------------------------------------------
def log_likelihood(w, X, y):
    """
    Compute log-likelihood for logistic regression.
    w: (d,) coefficient vector
    X: (N, d) design matrix
    y: (N,) labels in {-1, +1}
    """
    z = y * (X @ w)
    log_sigma = -np.log1p(np.exp(-z))
    return np.sum(log_sigma)

def log_prior(w, alpha=1.0):
    """
    Zero-mean Gaussian prior with precision alpha.
    """
    d = len(w)
    return -0.5 * alpha * np.dot(w, w) - 0.5 * d * np.log(2*np.pi/alpha)

def unnormalized_posterior(w, X, y, alpha=1.0):
    """Returns unnormalized posterior density: p(D|w)p(w)"""
    return np.exp(log_likelihood(w, X, y) + log_prior(w, alpha))

# ---------------------------------------------------------
# Integrands for Part 2
# ---------------------------------------------------------
# For 1D integration over w1 (with fixed w0 = -1)
def integrand_1d(w1_scalar, X, y):
    w = np.array([-1.0, w1_scalar])
    return unnormalized_posterior(w, X, y)

# Vectorized version of integrand_1d for array inputs
def integrand_1d_vec(w1_array, X, y):
    res = np.zeros_like(w1_array)
    for i, w1 in enumerate(w1_array):
        res[i] = integrand_1d(w1, X, y)
    return res

# For 2D integration over w0 and w1
def integrand_2d(w0_scalar, w1_scalar, X, y):
    w = np.array([w0_scalar, w1_scalar])
    return unnormalized_posterior(w, X, y)

# ---------------------------------------------------------
# 1D Numerical Integration Methods
# ---------------------------------------------------------
def trapezoidal_rule(f, a, b, tol, X, y):
    """
    Trapezoidal rule with iterative refinement and error estimation.
    f is the vectorized integrand function.
    """
    n = 1
    h = b - a
    I_old = (h / 2) * (f(np.array([a]), X, y)[0] + f(np.array([b]), X, y)[0])
    evals = 2
    
    for i in range(1, 20):  # max iterations
        h = h / 2
        # Calculate sum of new points
        x_new = a + h * np.arange(1, 2*n, 2)
        evals += len(x_new)
        I_new = 0.5 * I_old + h * np.sum(f(x_new, X, y))
        
        # Error estimation: E = (I_new - I_old) / 3
        err = np.abs((I_new - I_old) / 3)
        if i > 1 and err / np.abs(I_new) < tol:
            return I_new, evals, err
            
        I_old = I_new
        n *= 2
        
    return I_new, evals, err

def romberg_integration(f, a, b, tol, X, y):
    """
    Romberg integration.
    """
    max_k = 20
    R = np.zeros((max_k, max_k))
    
    n = 1
    h = b - a
    R[0, 0] = (h / 2) * (f(np.array([a]), X, y)[0] + f(np.array([b]), X, y)[0])
    evals = 2
    
    for k in range(1, max_k):
        h = h / 2
        x_new = a + h * np.arange(1, 2*n, 2)
        evals += len(x_new)
        R[k, 0] = 0.5 * R[k-1, 0] + h * np.sum(f(x_new, X, y))
        
        for j in range(1, k + 1):
            R[k, j] = R[k, j-1] + (R[k, j-1] - R[k-1, j-1]) / (4**j - 1)
            
        # Error estimation: E = |R[k,k] - R[k,k-1]|
        err = np.abs(R[k, k] - R[k, k-1])
        if k > 1 and err / np.abs(R[k, k]) < tol:
            return R[k, k], evals, err
            
        n *= 2
        
    return R[max_k-1, max_k-1], evals, err

# ---------------------------------------------------------
# 2D Numerical Integration Methods
# ---------------------------------------------------------
def simpson_2d(f, a, b, c, d, nx, ny, X, y):
    """
    2D Simpson's rule for double integral.
    f: integrand(w0, w1)
    [a,b]: limits for w0
    [c,d]: limits for w1
    nx, ny: number of intervals (must be even)
    """
    if nx % 2 != 0: nx += 1
    if ny % 2 != 0: ny += 1
        
    hx = (b - a) / nx
    hy = (d - c) / ny
    
    x = np.linspace(a, b, nx + 1)
    y_vals = np.linspace(c, d, ny + 1)
    
    # Weight matrices for Simpson's rule
    wx = np.ones(nx + 1); wx[1:-1:2] = 4; wx[2:-2:2] = 2
    wy = np.ones(ny + 1); wy[1:-1:2] = 4; wy[2:-2:2] = 2
    W = np.outer(wx, wy)
    
    Z = np.zeros((nx + 1, ny + 1))
    evals = 0
    for i in range(nx + 1):
        for j in range(ny + 1):
            Z[i, j] = f(x[i], y_vals[j], X, y)
            evals += 1
            
    I = (hx * hy / 9) * np.sum(W * Z)
    return I, evals

def simpson_2d_adaptive(f, L, tol, X, y):
    """Iteratively apply Simpson's 2D rule until tolerance is met."""
    n = 10
    I_old, evals_total = simpson_2d(f, -L, L, -L, L, n, n, X, y)
    
    for _ in range(5):
        n *= 2
        I_new, evals = simpson_2d(f, -L, L, -L, L, n, n, X, y)
        evals_total += evals
        
        # Error estimation for Simpson's is typically (I_new - I_old)/15 
        # but here we use a conservative relative error check
        err = np.abs((I_new - I_old) / 15)
        if err / np.abs(I_new) < tol:
            return I_new, evals_total, err
            
        I_old = I_new
        
    return I_new, evals_total, err

def crude_monte_carlo(num_samples, X, y):
    """
    Crude Monte Carlo using samples from the prior N(0, I).
    Integral = E_{w~p(w)} [p(D|w)] \approx 1/N * sum(p(D|w_i))
    """
    samples = np.random.randn(num_samples, 2)
    
    likelihoods = np.zeros(num_samples)
    for i in range(num_samples):
        # Compute likelihood p(D|w) = exp(log_likelihood)
        likelihoods[i] = np.exp(log_likelihood(samples[i], X, y))
        
    I_mc = np.mean(likelihoods)
    # Variance of the estimator is Var(likelihoods) / N
    var_mc = np.var(likelihoods) / num_samples
    err_mc = np.sqrt(var_mc) # Standard error
    
    return I_mc, err_mc

if __name__ == "__main__":
    np.random.seed(42)
    
    # ---------------------------------------------------------
    # (a) Data Generation
    # ---------------------------------------------------------
    N = 500
    w_true = np.array([-1, 1])
    x1 = np.random.randn(N) + 1  # x_i ~ N(1, 1)
    X = np.column_stack((np.ones(N), x1)) # Add bias term
    
    epsilon = np.random.normal(0, 2, N) # e_i ~ N(0, 4) -> SD = 2
    y = np.sign(X @ w_true + epsilon)
    y[y == 0] = 1 # Handle rare exact 0 case
    
    print("Data generated. N = 500.")
    print("---------------------------------------------------------")
    
    # ---------------------------------------------------------
    # 1D Integrations (Fixed w0 = -1)
    # ---------------------------------------------------------
    print("Part 2 (b & c): 1D Integration over w1 (w0 = -1)")
    # Using truncated interval [-L, L] due to Gaussian prior decay
    L_1d = 6.0 
    tol = 1e-6
    
    I_trapz, evals_trapz, err_trapz = trapezoidal_rule(integrand_1d_vec, -L_1d, L_1d, tol, X, y)
    print(f"Trapezoidal Rule:")
    print(f"  Integral value : {I_trapz:.10e}")
    print(f"  Est. Error     : {err_trapz:.10e}")
    print(f"  Evaluations    : {evals_trapz}")
    
    I_romb, evals_romb, err_romb = romberg_integration(integrand_1d_vec, -L_1d, L_1d, tol, X, y)
    print(f"Romberg Integration:")
    print(f"  Integral value : {I_romb:.10e}")
    print(f"  Est. Error     : {err_romb:.10e}")
    print(f"  Evaluations    : {evals_romb}")
    print("---------------------------------------------------------")
    
    # ---------------------------------------------------------
    # 2D Integrations
    # ---------------------------------------------------------
    print("Part 2 (d & e): 2D Integration over w0 and w1")
    L_2d = 6.0
    
    I_simp, evals_simp, err_simp = simpson_2d_adaptive(integrand_2d, L_2d, tol, X, y)
    print(f"Simpson's 2D Rule:")
    print(f"  Integral value : {I_simp:.10e}")
    print(f"  Est. Error     : {err_simp:.10e}")
    print(f"  Evaluations    : {evals_simp}")
    
    mc_samples = 1000000
    I_mc, err_mc = crude_monte_carlo(mc_samples, X, y)
    print(f"Crude Monte Carlo (Samples = {mc_samples}):")
    print(f"  Integral value : {I_mc:.10e}")
    print(f"  Std. Error     : {err_mc:.10e}")
    print("---------------------------------------------------------")
