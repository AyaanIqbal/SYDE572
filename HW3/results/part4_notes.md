For the centered data matrix X_c, the squared Frobenius norm equals the sum of squared singular values. Covariance eigenvalues satisfy λ_j = s_j²/(n − 1), so the common factor cancels when computing the fraction of variance retained.

Explained fraction for k PCs = Σ(j ≤ k) λ_j / Σ(all j) λ_j = 1 − ‖X_c − X̂_c,k‖_F² / ‖X_c‖_F².

Two PCs retain 97.7685% of total variance. The total centered squared Frobenius norm is 681.3706 cm² and the two-PC residual is 15.204644 cm². Thus 1 − 15.204644/681.3706 = 0.977685. The reconstruction RMSE per feature entry is 0.159189 cm.

One PC is enough to explain at least 50%: PC1 alone explains 92.4619%. Two coordinates per flower halve the original coordinate count, with a shared mean and component vectors also needed for reconstruction. Retained variance does not guarantee classification accuracy.
