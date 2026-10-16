# Part C: single-feature polynomial regression

Only the second feature, ZN (proportion of residential land zoned for lots
over 25,000 sq.ft.), is used. The first 100 observations are training data
and the remaining 406 are test data. ZN is standardized using the training
mean and standard deviation; this same transformation is used for test
points and the curve grid. Plot axes retain original units. Each model
includes a bias and is fitted without regularization.

| Degree | Training RMSE ($1000) | Test RMSE ($1000) |
| --- | ---: | ---: |
| 1 | 5.695083 | 9.352257 |
| 3 | 5.664927 | 9.434054 |
| 7 | 5.500617 | 10.423300 |

- **Degree 1:** The fitted line shows a modest positive association between
  ZN and median house value. Many observations with the same ZN have very
  different values, which this single-feature model cannot explain.
- **Degree 3:** The curve bends upward at larger ZN values. Training error
  improves only slightly relative to degree 1, while test error increases.
  The added flexibility does not improve generalization in this experiment.
- **Degree 7:** The curve oscillates strongly, especially in regions with
  few training observations, such as between ZN values of about 30 and 75.
  Training error is the lowest of these three models, but test error is
  the highest. This is evidence of overfitting: the additional flexibility
  fits the training sample better while predicting the held-out data worse.

The three plots show all training/test observations and the complete fitted
curve over the observed ZN range, without clipping high polynomial values.
The degrees were chosen to illustrate low, intermediate, and high model
complexity; the test errors are reported for comparison, not for tuning a
final model.
