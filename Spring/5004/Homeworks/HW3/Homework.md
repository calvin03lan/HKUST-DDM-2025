# MSDM5004 Spring 2026
## Part II Assignment 1
**Due: 30 Apr 23:59**

---

### 1. Detecting periodicity
The file `sunspots.txt` contains the observed number of sunspots on the Sun for each month since January 1749. The file contains two columns of numbers, the first representing the month and the second being the sunspot number.

(a) Write a program that reads the data in the file and makes a graph of sunspots as a function of time. You should see that the number of sunspots has fluctuated on a regular cycle for as long as observations have been recorded. Make an estimate of the length of the cycle in months.

(b) Modify your program to calculate the Fourier transform of the sunspot data and then make a graph of the magnitude squared $|c_{k}|^{2}$ of the Fourier coefficients as a function of $k$ --- also called the power spectrum of the sunspot signal. You should see that there is a noticeable peak in the power spectrum at a nonzero value of $k$.

(c) The appearance of this peak tells us that there is one frequency in the Fourier series that has a higher amplitude than the others around it, which corresponds to the periodic wave you can see in the original data. Find the approximate value of $k$ to which the peak corresponds. What is the period with this value of $k$? You should find that the period corresponds roughly to the length of the cycle that you estimated in part (a).

This kind of Fourier analysis is a sensitive method for detecting periodicity in signals. Even in cases where it is not clear to the eye that there is a periodic component to a signal, it may still be possible to find one using a Fourier transform.

---

### 2. Fourier filtering and smoothing
The file `dow.txt` contains the daily closing value for each business day from late 2006 until the end of 2010 of the Dow Jones Industrial Average (the "Dow").

Write a program to do the following:
(a) Read in the data from `dow.txt` and plot them on a graph.
(b) Calculate the coefficients of the discrete Fourier transform (DFT) of the data.
(c) Now set all but the first 10% of the elements of this array to zero (i.e., set the last 90% to zero but keep the values of the first 10%).
(d) Calculate the inverse Fourier transform of the resulting array and plot it on the same graph as the original data. Comment on what you see. What is happening when you set the Fourier coefficients to zero?
(e) Modify your program so that it sets all but the first 2% of the coefficients to zero and run it again.

Another file called `dow2.txt` contains data from 2004 until 2008, where the value changed considerably from around 9000 to 14000.
(f) Plot the data from `dow2.txt`. Smooth the data by setting all but the first 2% of the Fourier coefficients to zero and inverting the transform. You should see large deviations at the beginning and end of the plot. These occur because the DFT requires the function to be periodic.
(g) Modify your program to repeat the same analysis using **discrete cosine transforms (DCT)**. Again discard all but the first 2% of the coefficients. You should see a significant improvement at the ends of the interval because the cosine transform does not force the value to be the same at both ends (though it does force the gradient to be zero).

---

### 3. Image Deblurring
The file `blur.txt` contains a $1024 \times 1024$ grid representing a blurry photo. The camera has a Gaussian point spread function (PSF) with width $\sigma=25$:
$$f(x,y)=\exp\left(-\frac{x^{2}+y^{2}}{2\sigma^{2}}\right)$$

(a) Read the values and draw a density plot. (Hint: sky is bright at the top, ground is dark at the bottom).
(b) Create an array of the same size containing the Gaussian PSF. Because it is treated as periodic, there should be bright patches in each of the four corners.
(c) Deconvolve the photo by:
1. Fourier transforming both the photo and the PSF.
2. Dividing the photo transform by the PSF transform. (If a PSF coefficient is $< 10^{-3}$, do not divide; leave it alone to avoid division by zero).
3. Performing an inverse transform to get the deblurred photo.

---

### 4. Solve the Poisson Equation
Solve the Poisson Equation:
$$\frac{\partial^{2}\phi}{\partial x^{2}}+\frac{\partial^{2}\phi}{\partial y^{2}}=\rho(x,y)$$
in $0 \le x \le 1, 0 \le y \le 2$ by the spectral method with $\Delta \le 0.01$.

**Boundary Conditions:**
* $\phi(x=0, 0 < y < 2) = 0$
* $\phi(x=1, 0 < y < 2) = 0$
* $\phi(0 \le x \le 1, y = 0) = 0$
* $\phi(0 \le x \le 1, y = 2) = \sqrt{1-x^{2}}$

**Source Density:**
$$\rho(x, y) = \begin{cases} 16\pi & 0.25 \le x < 0.5, 1.25 \le y < 1.5 \\ 0 & \text{Otherwise} \end{cases}$$

Show your answer by plotting $\phi(x,y)$.