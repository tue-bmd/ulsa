### Derivation for our estimate of the entropy for a 

We want to compute the entropy of $p(y_t \mid A^\ell, y_{<t})$ 

We can choose to model $p(x_t \mid y_{<t})$ as a pixelwise independent Gaussian, so that $p(x_t \mid y_{<t}) = \mathcal{N}(x_t; \bar{x}_t, S_{x_t})$ where $\bar{x}_t$ is the sample mean of the belief distribution (samples $x_t \mid y_{<t})$, and $S_{x_t}$ is the sample pixelwise variance, i.e. a diagonal covariance with entries $\sigma_{x_t, i}^2$ for pixels $i$.
 
Given that the measurement model $y_t = U(A^\ell)x_t$ is linear (where $U(A^\ell)$ makes a matrix with 1s on the diagonal at indices contained in $A^\ell$), we get that $p(y_t \mid A^\ell, y_{<t})$ is a simple transformation of $p(x_t \mid y_{<t})$:
$$
p(y_t \mid A^\ell, y_{<t}) = \mathcal{N}(y_t; U(A^\ell)\bar{x}_t, U(A^\ell)S_{x_t}U(A^\ell)^\top)
$$
The entropy of this distribution is then:
$$
H(y_t \mid A^\ell, y_{<t}) = \frac{1}{2}\log((2\pi e)^M |U(A^\ell)S_{x_t}U(A^\ell)^\top|)
$$
The matrix multiplications of $S_{x_t}$ have the effect of selecting only the variances of the pixels revealed by $A^\ell$. Given that the determinant of a diagonal matrix is just the product of its diagonal, the entropy simplifies to:
$$
H(y_t \mid A^\ell, y_{<t}) = \frac{1}{2}\log((2\pi e)^M \prod_{i \in A^\ell}\sigma_{x_t, i}^2)
$$

Bringing the product outside of the log to become a sum, we get finally:
$$
H(y_t \mid A^\ell, y_{<t}) = \sum_{i \in A^\ell}\frac{1}{2}\log((2\pi e)^{|A^\ell|} \sigma_{x_t, i}^2)
$$
In other words, the entropy of the measurement $y_t$ observed by acquiring line $\ell$ is equal to the sum of pixelwise entropies of the pixels revealed by $\ell$.