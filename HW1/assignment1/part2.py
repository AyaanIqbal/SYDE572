"""Fit a line or general parabola by analytical and sequential Newton methods."""
from pathlib import Path
import numpy as np

POINTS = np.array([(0, 0.5), (2, 3.5), (1, 1.5), (3, 7.5)])


def _data(points, power):
    """Build columns [x, 1] or [x², x, 1] for the chosen model."""
    data = np.asarray(points, dtype=float)
    if power not in (1, 2):
        raise ValueError('Choose power 1 (line) or 2 (parabola)')
    if data.ndim != 2 or data.shape[1] != 2 or len(data) < power+1:
        raise ValueError('Supply enough (x, y) points for the model')
    if not np.all(np.isfinite(data)):
        raise ValueError('Data must be finite')
    design = np.vander(data[:, 0], power+1)
    if np.linalg.matrix_rank(design) < power+1:
        raise ValueError('The parameters require distinct x values')
    names = ('m', 'b') if power == 1 else ('a', 'b', 'c')
    return design, data[:, 1], names


def _row(parameters, names, design, y):
    """Return named coefficients and their mean squared error."""
    row = dict(zip(names, map(float, parameters)))
    row['mse'] = float(np.mean((design @ parameters-y)**2))
    return row


def analytical_fit(points=POINTS, power=1):
    """Set the MSE derivatives to zero and solve the normal equations."""
    design, y, names = _data(points, power)
    parameters = np.linalg.solve(design.T @ design, design.T @ y)
    return _row(parameters, names, design, y)


def sequential_newton_fit(points=POINTS, power=1, initial=None,
                          tol=1e-10, max_iter=1000):
    """Update one parameter at a time, using the newest values each time.

    A sweep updates m then b, or a then b then c. Set max_iter=3 to
    reproduce the three handwritten iterations, starting from zeros.
    """
    design, y, names = _data(points, power)
    if not np.isfinite(tol) or tol <= 0 or not isinstance(max_iter, int) or max_iter < 0:
        raise ValueError('Require positive finite tol and nonnegative max_iter')
    parameters = np.zeros(len(names)) if initial is None else np.array(initial, dtype=float)
    if parameters.shape != (len(names),) or not np.all(np.isfinite(parameters)):
        raise ValueError('Supply one finite initial value for each parameter')
    curvature = 2*np.mean(design**2, axis=0)
    steps = []
    converged = False
    for iteration in range(max_iter+1):
        row = _row(parameters, names, design, y)
        steps.append(dict(iteration=iteration, **row))
        gradient = 2*design.T @ (design @ parameters-y)/len(y)
        if not np.all(np.isfinite(gradient)) or not np.isfinite(row['mse']):
            raise ValueError('Nonfinite fit evaluation')
        if np.max(np.abs(gradient)) <= tol:
            converged = True
            break
        if iteration == max_iter:
            break
        for j in range(len(names)):
            # Recompute residuals so each update uses the newest parameters.
            residual = design @ parameters-y
            derivative = 2*np.mean(residual*design[:, j])
            parameters[j] -= derivative/curvature[j]
    return dict(**row, steps=steps, converged=converged,
                status='converged' if converged else 'max_iter')


def plot_fit(points, power, result, exact, path):
    """Plot the first three sweeps and the analytical minimum."""
    import matplotlib.pyplot as plt
    x, y = np.asarray(points).T
    xs = np.linspace(x.min()-.3, x.max()+.3, 400)
    names = ('m', 'b') if power == 1 else ('a', 'b', 'c')
    fig, (ax, loss) = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for row in result['steps'][1:4]:
        ax.plot(xs, np.polyval([row[k] for k in names], xs), '--',
                label=f'Iteration {row["iteration"]}')
    ax.plot(xs, np.polyval([exact[k] for k in names], xs), color='teal',
            label='Analytical minimum')
    ax.scatter(x, y, color='black', label='Data', zorder=5)
    ax.set(xlabel='x', ylabel='y', title='Line' if power == 1 else 'Parabola: ax² + bx + c')
    ax.legend(fontsize=8)
    rows = result['steps'][:4]
    loss.plot([s['iteration'] for s in rows], [s['mse'] for s in rows], 'o-', color='navy')
    loss.axhline(exact['mse'], color='teal', linestyle='--', label='Analytical MSE')
    loss.set(xlabel='Iteration', ylabel='MSE', xticks=range(4), title='First three iterations')
    loss.legend(fontsize=8)
    for axis in (ax, loss):
        axis.grid(alpha=.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    output = Path(__file__).parent / 'plots'
    output.mkdir(exist_ok=True)
    for name, power in [('line', 1), ('parabola', 2)]:
        exact = analytical_fit(power=power)
        numerical = sequential_newton_fit(power=power, max_iter=3)
        print(f'{name}: analytical={exact}')
        for row in numerical['steps'][1:]:
            print(row)
        plot_fit(POINTS, power, numerical, exact, output / f'part2_{name}.png')


if __name__ == '__main__':
    main()
