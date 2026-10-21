# Part B: degree-1 weights

Coefficients predict house value in $1000. Normalized weights correspond to
one training standard deviation; original-unit weights correspond to one
unit of the original feature. The bias is included below.

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

**RM (6th feature):** One additional average room predicts about
$8,591 higher value, holding other features fixed. The
positive coefficient is reasonable because larger homes generally cost more.

**DIS (8th feature):** One additional distance unit predicts about
$7.63 higher value, holding other features fixed. The
positive sign is less intuitive if proximity to employment increases value,
but the effect is tiny and may reflect correlated features and the limited
training sample. These associations are not causal conclusions.

CHAS is constant in the training data, so its zero coefficient is not evidence
of no effect; its influence cannot be estimated from this subset.
