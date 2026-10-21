# Part D: regularized polynomial regression

A degree-8 polynomial is fitted using only ZN (the second feature) and the
first 100 observations. Ten-fold cross-validation splits a seeded random
permutation (seed 572) into ten groups of ten. Each fit uses 90 training rows
and ten validation rows. The same folds are used for every lambda. Input
normalization is fitted on each fold's training rows before taking powers;
the target stays in $1000. The other 406 observations are not used here.

The objective is sum((y - Phi w)^2) + lambda * sum(w[1:]^2), with an
unpenalized bias. Lambda therefore multiplies the penalty relative to the
sum, rather than the mean, of squared residuals. Augmented least squares
solves this objective without explicitly inverting the normal equations.

| Lambda | Mean validation RMSE ($1000) |
| --- | ---: |
| 0 | 9.984825 |
| 0.01 | 6.586052 |
| 0.1 | 7.673666 |
| 1 | 9.343756 |
| 10 | 10.251101 |
| 100 | 9.792464 |
| 1000 | 7.422387 |

Choose **lambda = 0.01**, which has the lowest average validation RMSE
(6.586052) among the seven candidates. The unregularized baseline is
9.984825. Regularization constrains polynomial coefficients and can
reduce sensitivity to the training sample. The plot compares the measured
validation errors; lambda = 0 is shown as a horizontal baseline because zero
cannot be placed on a logarithmic x-axis. The selected value is specific to
this normalization, objective convention, and fold split.
