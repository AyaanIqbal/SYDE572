# Part B: degree-1 weights

The model uses the first 100 observations, all 13 features, and a bias.
Features are standardized as z_j = (x_j - mean_j) / std_j, using training
statistics. The target is not normalized and is measured in $1000.

| Feature | Normalized weight | Original-unit weight |
| --- | ---: | ---: |
| Bias | 22.309000 | -37.904200 |
| CRIM | -1.337974 | -3.455946 |
| ZN | 0.122993 | 0.005382 |
| INDUS | -0.531065 | -0.168779 |
| CHAS | 0.000000 | 0.000000 |
| NOX | 1.002651 | 21.515396 |
| RM | 4.195634 | 8.590961 |
| AGE | -2.166275 | -0.079237 |
| DIS | 0.010715 | 0.007627 |
| RAD | -0.193778 | -0.139906 |
| TAX | -0.348631 | -0.007630 |
| PTRATIO | 0.261264 | 0.154556 |
| B | 0.181073 | 0.006645 |
| LSTAT | 0.148761 | 0.026230 |

The normalized model is y_hat = w_0 + sum(w_j z_j). Each normalized feature
weight describes the predicted change in $1000 for a one-training-standard-
deviation increase, holding other features fixed. To interpret changes in
original units, use beta_j = w_j / std_j and
beta_0 = w_0 - sum(beta_j mean_j). These two forms give identical predictions.

## Rooms per dwelling (RM)

The normalized weight is 4.195634. The training standard deviation
is 0.488378 rooms, so the original-unit coefficient is
8.590961 ($1000 per room). Holding all other features fixed, one more
room per dwelling increases the predicted median house value by approximately
$8,591. The positive sign is reasonable: larger homes would
generally be expected to have higher values. This is a fitted association,
not a causal estimate.

## Distance to employment centres (DIS)

The normalized weight is 0.010715. The training standard deviation
is 1.404998 distance units, so the original-unit coefficient
is 0.007627 ($1000 per dataset distance unit). Holding other features
fixed, a one-unit increase in weighted distance increases predicted value by
approximately $7.63. This effect is very small compared
with RM: a one-standard-deviation increase changes the prediction by only
$10.72. Its positive sign may initially seem
surprising if proximity to employment is expected to increase value. However,
this coefficient describes a conditional association in the first 100 rows;
correlated features and the restricted training sample can affect its sign.
The fit alone does not establish why this association occurs.

## Indexing and limitations

RM and DIS are Python feature indices 5 and 7 (CSV feature positions 6 and 8).
With the bias first, they are entries 6 and 8 in the weight vector. Identify
them by name. These are the 6th and 8th features specified in the instructor's
assignment correction; the corrected numbering does not change this fit.

CHAS is zero throughout the training subset. Its coefficient is therefore
unidentifiable from these observations; the least-squares solver returns zero
in its minimum-norm solution. This is not evidence that CHAS has no effect.
The model has training RMSE 2.037201 and test RMSE
36.444950 (both in $1000), so interpretation of coefficient
magnitudes should account for its poor generalization to the test subset.
