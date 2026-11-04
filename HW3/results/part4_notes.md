For centered data X_c, the squared Frobenius norm equals the sum of squared singular values. Since covariance eigenvalues satisfy λ_j = s_j²/(n − 1), the fraction of variance retained can also be calculated from reconstruction error:

Explained fraction for k PCs = Σ(j ≤ k) λ_j / Σ(all j) λ_j = 1 − ‖X_c − X̂_c,k‖_F² / ‖X_c‖_F².

The first two PCs explain 97.7685% of total variance. The Frobenius-norm calculation confirms this: 1 - 15.204644/681.3706 = 0.977685, where the numerator is the two-PC squared reconstruction error and the denominator is the total centered squared norm.

The 50% target is reached with a single component: PC1 captures 92.4619% of the dataset’s variance.
