# Part B: degree-1 weights

Coefficients predict house value in $1000. Normalized weights correspond to
one training standard deviation; original-unit weights correspond to one
unit of the original feature.

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

## Key coefficients

| Variable | Normalized-model coefficient | Original-unit coefficient |
| --- | ---: | ---: |
| Bias | 22.309000 | -37.904200 |
| Rooms (RM, 6th feature) | 4.195634 | 8.590961 |
| Distance (DIS, 8th feature) | 0.010715 | 0.007627 |

**Bias:** The normalized-model bias predicts $22,309 when
all features equal their training means. This is a reasonable baseline: it
equals the average training house value because the inputs are centered.

**Rooms (RM): Reasonable.** Holding other features fixed, one additional
average room predicts about $8,591 higher value. The positive
coefficient agrees with the expectation that larger homes generally cost more.

**Distance (DIS): The positive sign is contrary to the expected relationship.**
Holding other features fixed, one additional distance unit predicts only
$7.63 higher value. If proximity to employment increases
value, a negative coefficient would be more intuitive. However, the effect
is nearly zero, so this is weak evidence of a positive relationship; correlated
features and the limited training sample may explain the unexpected sign.

CHAS is constant in the training data, so its zero coefficient is not
evidence of no effect.
