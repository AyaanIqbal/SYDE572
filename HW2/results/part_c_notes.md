# Part C: single-feature polynomial regression

Using only ZN (the second feature):

| Degree | Training RMSE ($1000) | Test RMSE ($1000) |
| --- | ---: | ---: |
| 1 | 5.695083 | 9.352257 |
| 3 | 5.664927 | 9.434054 |
| 7 | 5.500617 | 10.423300 |

Degree 1 gives a straight line; degree 3 adds modest curvature with little
training improvement. Degree 7 oscillates strongly between sparse training
points. As degree increases, training error falls while test error rises,
showing overfitting.
