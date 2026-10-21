# Part E: Gaussian basis regression

All 13 features are used, with the first 100 observations for training and
the remaining 406 for testing. Features are standardized using training-only
means and standard deviations. Zero standard deviations use a scale of one.
The target remains in $1000. No regularization is applied.

Each model uses K Gaussian functions plus a separate bias:
phi_j(z) = exp(-||z - centre_j||^2 / (2 sigma^2)). The assignment does not
specify a width; sigma = 1 in standardized feature space is used throughout,
without tuning it against test errors. Results depend on this width choice.
A seeded random permutation (seed 572) of the 100 training rows is generated;
the first K rows supply the centres. These are nested sets as K increases.
Weights are fitted using unregularized least squares.

| Gaussian basis functions | Training RMSE ($1000) | Test RMSE ($1000) |
| --- | ---: | ---: |
| 5 | 5.193195 | 9.847776 |
| 15 | 4.292959 | 9.792367 |
| 25 | 3.846677 | 9.810629 |
| 35 | 3.663275 | 9.913723 |
| 45 | 3.244222 | 9.816564 |
| 55 | 3.061174 | 9.863992 |
| 65 | 2.858291 | 9.845859 |
| 75 | 1.907602 | 11.935474 |
| 85 | 0.763893 | 11.798226 |
| 95 | 0.298743 | 11.754598 |

Training RMSE falls from 5.193 at K = 5 to 0.299
at K = 95 as additional basis functions increase model flexibility. Test
RMSE stays around 9.8 through K = 65, then jumps to about 11.9 at K = 75
and remains above 11.7. The widening gap is evidence of overfitting.
Test error does not have to decrease with training error: the centres cover only
the first 100 observations, and a flexible model may generalize poorly to
the other observations. Gaussian functions are local, so predictions far
from every centre approach the fitted bias. Interpret the plotted test
curve as held-out evaluation, rather than using it to tune the width or
centre selection. Part F will use cross-validation for regularization.
