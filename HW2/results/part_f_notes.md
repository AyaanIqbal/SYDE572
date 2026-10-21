# Part F: regularized Gaussian regression

All 13 features and the first 100 observations are used. The model has 90
Gaussian basis functions plus an unpenalized bias. As in Part E, sigma = 1
in standardized feature space; it is fixed rather than tuned on the test set.
The objective is summed squared residuals + lambda * sum(w[1:]^2).
Augmented least squares fits the model without explicit matrix inversion.

Ten-fold cross-validation uses a seeded permutation (seed 572), with 90
training rows and ten validation rows per fold. Normalization is learned
only from those 90 training rows. All 90 become centres, so no validation
row is used as a centre. Folds and centres are identical across lambda
candidates. The remaining 406 observations are not used to choose lambda.
The reported metric is the arithmetic mean of the ten fold RMSE values.

| Lambda | Mean validation RMSE ($1000) |
| --- | ---: |
| 0 | 3.999131 |
| 0.01 | 3.840319 |
| 0.1 | 3.730653 |
| 1 | 4.026220 |
| 10 | 4.998234 |
| 100 | 5.635366 |
| 1000 | 5.789236 |

Choose **lambda = 0.1**, which minimizes average validation RMSE
(3.730653) among the specified candidates. The unregularized
baseline is 3.999131. The plot uses a logarithmic lambda axis and
shows lambda = 0 as a horizontal baseline. Regularization limits coefficient
magnitudes and can reduce overfitting; very strong penalties suppress the
Gaussian contributions and move the model toward a constant prediction.
This selection depends on the fixed Gaussian width, normalization, fold
split, and summed-error convention for lambda.

The selected model is refitted on all 100 training observations, selecting
90 centres from their seeded permutation, and saved in part_f_model.npz.
