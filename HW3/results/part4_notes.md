For the centered data matrix X_c, the squared Frobenius norm equals the sum of squared singular values. Since the covariance eigenvalues satisfy λ_j = s_j²/(n − 1), the common factor cancels when calculating the fraction of variance retained. This connects explained variance to reconstruction error:

Explained fraction for k PCs = Σ(j ≤ k) λ_j / Σ(all j) λ_j = 1 − ‖X_c − X̂_c,k‖_F² / ‖X_c‖_F².

The first two PCs retain 97.7685% of total variance. Using the total centered squared Frobenius norm of 681.3706 cm² and the two-PC residual of 15.204644 cm² gives 1 − 15.204644/681.3706 = 0.977685, confirming the same result from reconstruction error. Across all feature entries, the reconstruction RMSE is 0.159189 cm.

Only one PC is needed to explain at least 50% of the variance because PC1 already accounts for 92.4619%. Keeping two coordinates per flower halves the original coordinate count, although reconstruction also requires the shared mean and component vectors. This makes PCA useful for compression here, but the amount of variance retained does not by itself tell us how accurately the species can be classified.
