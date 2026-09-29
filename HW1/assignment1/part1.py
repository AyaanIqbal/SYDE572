"""Manual closest-point searches. Run this file to generate the demonstrations."""
from math import isfinite, sqrt
from pathlib import Path
import json


def _validate(tol, max_iter):
    if not isfinite(tol) or tol <= 0:
        raise ValueError('tol must be positive and finite')
    if not isinstance(max_iter, int) or max_iter < 0:
        raise ValueError('max_iter must be a nonnegative integer')


def _finite(*values):
    if not all(isfinite(v) for v in values):
        raise ValueError('Inputs and function evaluations must be finite')


def _result(f, x, x0, y0, history, status):
    y = float(f(x))
    distance = sqrt((x - x0)**2 + (y - y0)**2)
    _finite(x, y, distance)
    return dict(x=x, point=(x, y), distance=distance, history=history,
                converged=status == 'converged', status=status)


def newton_closest_point(f, df, ddf, x0, y0, initial_guess,
                         tol=1e-10, max_iter=100, curvature_tol=1e-12):
    """Find a local distance minimum; arbitrary starts need not converge.

    History includes the initial estimate. A stationary point with nonpositive
    curvature is not reported as a converged minimum. No global guarantee.
    """
    _validate(tol, max_iter)
    _finite(x0, y0, initial_guess, curvature_tol)
    if curvature_tol <= 0:
        raise ValueError('curvature_tol must be positive')
    x = float(initial_guess)
    history = []
    status = 'max_iter'
    for iteration in range(max_iter + 1):
        y, slope, curvature = float(f(x)), float(df(x)), float(ddf(x))
        d1 = 2*(x-x0) + 2*(y-y0)*slope
        d2 = 2 + 2*slope**2 + 2*(y-y0)*curvature
        distance_squared = (x-x0)**2 + (y-y0)**2
        _finite(y, d1, d2, distance_squared)
        history.append(dict(iteration=iteration, x=x, D=distance_squared,
                            gradient=d1, curvature=d2))
        if abs(d2) <= curvature_tol:
            status = 'near_zero_curvature'
            break
        if abs(d1) <= tol:
            status = 'converged' if d2 > 0 else 'not_a_minimum'
            break
        if iteration == max_iter:
            break
        x -= d1/d2
        _finite(x)
    return _result(f, x, x0, y0, history, status)


def golden_section_closest_point(f, x0, y0, a, b, tol=1e-8, max_iter=200):
    """Minimize squared distance on [a,b], assuming it is unimodal there.

    tol bounds interval width, not floating-point accuracy of the minimizer.
    History stores the initial interval and each subsequent interval.
    """
    _validate(tol, max_iter)
    _finite(x0, y0, a, b)
    if a >= b:
        raise ValueError('Require a < b')
    def objective(x):
        value = (x-x0)**2 + (float(f(x))-y0)**2
        _finite(value)
        return value
    r = 2 - (1 + sqrt(5))/2
    x1, x2 = a + r*(b-a), b - r*(b-a)
    d1, d2 = objective(x1), objective(x2)
    history = []
    for iteration in range(max_iter + 1):
        history.append(dict(iteration=iteration, a=a, b=b, x1=x1, x2=x2,
                            D1=d1, D2=d2))
        if b-a <= tol or iteration == max_iter:
            break
        if d1 < d2:
            b, x2, d2 = x2, x1, d1
            x1 = a + r*(b-a)
            d1 = objective(x1)
        else:
            a, x1, d1 = x1, x2, d2
            x2 = b - r*(b-a)
            d2 = objective(x2)
    status = 'converged' if b-a <= tol else 'max_iter'
    return _result(f, (a+b)/2, x0, y0, history, status)


def plot_search(f, point, newton, golden, bounds, title, path):
    import numpy as np
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    ax, intervals = axes
    xs = np.linspace(*bounds, 600)
    ax.plot(xs, [f(x) for x in xs], label=title, color='navy')
    ax.scatter(*point, marker='*', s=130, color='black', label='External point', zorder=5)
    nx = [h['x'] for h in newton['history']]
    ax.plot(nx, [f(x) for x in nx], 'o--', color='darkorange',
            alpha=0.7, label='Newton estimates', markersize=5)
    q = newton['point']
    ax.plot([point[0], q[0]], [point[1], q[1]], color='gray', label='Shortest segment')
    ax.scatter(*q, s=90, color='crimson', label='Newton final', zorder=6)
    ax.scatter(*golden['point'], s=130, facecolors='none', edgecolors='green',
               label='Golden final', zorder=7)
    ax.set(xlabel='x', ylabel='y', title=f'Closest point to {point}')
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(fontsize=8)
    for h in golden['history']:
        intervals.plot([h['a'], h['b']], [h['iteration']]*2, color='teal', linewidth=2)
    intervals.axvline(golden['x'], color='crimson', linestyle='--', label='Final x')
    intervals.set(xlabel='Search interval in x', ylabel='Iteration', title='Golden Section interval evolution')
    intervals.invert_yaxis()
    intervals.legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    from math import exp
    output = Path(__file__).parent / 'plots'
    output.mkdir(exist_ok=True)
    cases = [(f'parabola_{p}', lambda x:x*x+5, lambda x:2*x, lambda x:2,
              (p, 0), 1.0, (-2, 2), (-2, 2), 'y = x² + 5')
             for p in (0, -4, -8, 2, 6)]
    cases += [('cubic', lambda x:x**3-2*x+1, lambda x:3*x*x-2, lambda x:6*x,
               (0, 0), 0.2, (0, 0.7), (-0.5, 1.2), 'y = x³ − 2x + 1'),
              ('exponential', exp, exp, exp, (0, 0), 0.0,
               (-1, 0), (-1.5, 0.8), 'y = exp(x)')]
    results = {}
    for name, f, df, ddf, point, guess, interval, bounds, title in cases:
        n = newton_closest_point(f, df, ddf, *point, guess)
        g = golden_section_closest_point(f, *point, *interval)
        assert n['converged'] and g['converged'], (name, n['status'], g['status'])
        assert abs(n['x']-g['x']) < 1e-6, name
        results[name] = dict(newton=n, golden=g)
        print(f'{name:14s} Newton x={n["x"]: .9f}  Golden x={g["x"]: .9f}  distance={n["distance"]:.9f}')
        plot_search(f, point, n, g, bounds, title, output / f'part1_{name}.png')
    (output / 'part1_history.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
