# Homework 3 Correction List

## Question 1: Detecting Periodicity
* **Error:** The time-domain cycle estimate (approx. 4.00 months) is incorrect.
* **Reason:** The peak detection logic picks up high-frequency noise and small fluctuations in the raw data rather than the decadal solar cycle.
* **Correction:** Apply a smoothing filter (e.g., a moving average) to the data before detecting local maxima, or estimate the period by visual inspection of the plot (should be approx. 130–132 months).

## Question 4: Poisson Equation
* **Error:** The boundary lifting function $g(x,y) = \frac{y}{2}\sqrt{1-x^2}$ is incorrect.
* **Reason:** It violates the required boundary condition $\phi(0, y) = 0$ because $g(0, y) = y/2 \neq 0$. This prevents the resulting $u(x,y)$ from being homogeneous on all sides, which is a requirement for the Discrete Sine Transform (DST) method.
* **Correction:** Define a lifting function $g(x,y)$ that satisfies $g=0$ at $x=0$, $x=1$, and $y=0$, while maintaining $g = \sqrt{1-x^2}$ at $y=2$. For example, ensure the $x$-dependent part vanishes at $x=0$ (note that $\sqrt{1-x^2}$ is 1 at $x=0$, not 0).