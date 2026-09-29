# SYDE 572 Assignment 1 — Python

From the `HW1` directory (`cd HW1` from the repository root):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
MPLBACKEND=Agg MPLCONFIGDIR=.mplconfig .venv/bin/python assignment1/part1.py
MPLBACKEND=Agg MPLCONFIGDIR=.mplconfig .venv/bin/python assignment1/part2.py
.venv/bin/python -m unittest discover -s assignment1 -v
```

`plots/` contains seven Part 1 figures, two Part 2 figures, and JSON files
with every iteration. Running the scripts regenerates these outputs. The
algorithms use no SciPy optimizers. Import either module to reuse its functions;
importing does not run demonstrations.

## Closest points

`newton_closest_point(f, df, ddf, x0, y0, initial_guess, tol, max_iter)`
applies Newton to D'(x), with D=(x-x0)²+(f(x)-y0)². Its tolerance is on
|D'|; it stops and reports status if curvature is nearly zero, the iteration
limit is reached, or the stationary point has negative curvature.

`golden_section_closest_point(f, x0, y0, a, b, tol, max_iter)` reuses one
interior evaluation per interval update. Its tolerance is on interval width.
Both return dictionaries containing `x`, `point`, `distance`, `history`,
`converged`, and `status`. Inspect status before interpreting an estimate as
successful. Invalid/nonfinite evaluations raise ValueError (extreme arithmetic
may also raise OverflowError).

Newton depends on the initial guess; even a local minimum need not be global.
Golden Section needs a suitable unimodal distance objective on the supplied
interval, not merely a unimodal f. The required parabola uses [-2,2] and
initial guess 1. Here D''=12x²+22>0, so each stationary point is the unique
global minimum. All five minima lie in the interval.

The cubic demonstration uses point (0,0), guess 0.2, and interval [0,0.7];
the exponential y=exp(x) uses point (0,0), guess 0, and [-1,0]. Its
first and second derivatives are both exp(x), and its squared-distance
curvature is D''=2+4 exp(2x)>0. These intervals select
unimodal distance objectives. Plot labels show Newton estimates and Golden
Section intervals, including their initial states.

Floating-point cancellation in squared distances limits Golden Section x
accuracy to roughly 10⁻⁸–10⁻⁷ here, even with a narrow final interval. The
comparison therefore uses 10⁻⁶ agreement, rather than equating bracket width
with guaranteed minimizer accuracy. In particular its x at (0,0) may be a
few times 10⁻⁸ rather than exactly zero; the distance is still 5.

## Fitting

The parabola is explicitly **y = ax² + b**, following the handoff. It is not
a three-parameter quadratic. Both models share the form c z+b, where z=x
for the line and z=x² for the parabola. Functions accept alternative data,
initial parameters, tolerances, and iteration limits.

`analytical_fit` solves the two normal equations by elimination:
c=Cov(z,y)/Var(z), b=mean(y)-c mean(z). `sequential_newton_fit` uses
∂MSE/∂c=2 mean((cz+b-y)z), ∂²MSE/∂c²=2 mean(z²), and
∂MSE/∂b=2 mean(cz+b-y), ∂²MSE/∂b²=2. It updates c first,
then evaluates the b derivative using the new c. For the supplied points,
these are exactly the derivatives in the handoff. Each history row records
one full sweep, including the initial (0,0). Convergence requires both
partial derivatives to be within tolerance.

Analytical values:

| Model | coefficient | intercept b | MSE |
| --- | ---: | ---: | ---: |
| Line | 2.3 | -0.2 | 0.575 |
| ax²+b | 0.765306122449 | 0.571428571429 | 0.012755102041 |

The verification checks include independent NumPy least squares, the supplied
early parabola sweeps, decreasing MSE, distance stationarity, interval
contraction, and failure statuses. No mathematical discrepancies were found
in the supplied formulas or expected results.
