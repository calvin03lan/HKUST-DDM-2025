**You are required to submit a report and all the codes for a project. Don't save the result of every time step in the evolution.**

1. Consider the problem

   $$
   \begin{cases} 
   \frac{\partial u}{\partial t} = a \frac{\partial^2 u}{\partial x^2} & -1 < x < 1, \quad t > 0 \\
   u(-1, t) = u(1, t) = 0 & t > 0 \\
   u(x, 0) = u_0(x) & -1 \le x \le 1
   \end{cases}
   $$

   where $a = 2$ and

   $$
   u_0(x) = 
   \begin{cases} 
   \frac{1}{3}(x + 1) & -1 \le x \le \frac{1}{2} \\
   1 - x & \frac{1}{2} < x \le 1.
   \end{cases}
   $$

   (1) Obtain numerical solution using the explicit scheme

   $$U_j^{n+1} = U_j^n + a\mu(U_{j+1}^n - 2U_j^n + U_{j-1}^n),$$

   where $\mu = \frac{\Delta t}{(\Delta x)^2}$. Use $J = 20$, $\Delta x = 0.1$, and (i) $\Delta t = 0.0025$ (ii) $\Delta t = 0.0026$. Plot the numerical solution at $t = 0, \Delta t, 25\Delta t, 50\Delta t$.

   (2) Obtain numerical solution using the Crank-Nicolson method. You are required to use the Thomas algorithm to solve the linear system in the numerical discretization.
   Use $J = 20$, $\Delta x = 0.1$, and (i) $\Delta t = 0.0025$ (ii) $\Delta t = 0.0026$. Plot the numerical solution at $t = 0, \Delta t, 25\Delta t, 50\Delta t$.

2. Consider the equation $u_t + 1.8u_x = 0$ in $x \in (-2, 2)$. Assume that the initial condition is
   $$
   u_0(x) = 
   \begin{cases} 
   1 & x \le 0 \\
   0 & x > 0 
   \end{cases}
   $$

   Use the boundary conditions $u(-2, t) = 1$ and $u(2, t) = 0$ (The latter is not valid after the jump moves out of the simulation region, but we are not going to compute the solution after that time). Choose $\nu = \Delta t/\Delta x = 0.25$. Compute and plot the solution at time $t = 0.5$, using

   (1) upwind method, $\Delta t = 0.01$ and $\Delta t = 0.0025$,
   (2) Lax-Wendroff method, $\Delta t = 0.01$ and $\Delta t = 0.0025$.
   (3) Compare the results with the exact solution and discuss the behavior of these numerical solutions.

   **Remark:** I would like to see four figures (two methods and two $\Delta t$'s), with the exact solution plotted in each figure.