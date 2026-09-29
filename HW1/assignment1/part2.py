"""Analytical and sequential parameter-wise Newton least-squares fits."""
from pathlib import Path
import json
import numpy as np

POINTS = np.array([(0, 0.5), (2, 3.5), (1, 1.5), (3, 7.5)])


def _data(points, power):
    data = np.asarray(points, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2 or len(data) < 2:
        raise ValueError('Supply at least two (x,y) points')
    if power not in (1, 2) or not np.all(np.isfinite(data)):
        raise ValueError('Require finite data and power 1 (line) or 2 (parabola)')
    z, y = data[:, 0]**power, data[:, 1]
    if not np.all(np.isfinite(z)) or np.var(z) <= 1e-14:
        raise ValueError('The chosen model needs varying x**power values')
    return z, y


def analytical_fit(points=POINTS, power=1):
    """Solve the two normal equations for y = coefficient*x**power + b.

    Setting both MSE derivatives to zero gives coefficient=Cov(z,y)/Var(z)
    and b=mean(y)-coefficient*mean(z), where z=x**power.
    """
    z, y = _data(points, power)
    coefficient = float(np.mean((z-z.mean())*(y-y.mean())) / np.var(z))
    b = float(y.mean() - coefficient*z.mean())
    return dict(coefficient=coefficient, b=b,
                mse=float(np.mean((coefficient*z+b-y)**2)))


def sequential_newton_fit(points=POINTS, power=1, initial=(0, 0),
                          tol=1e-10, max_iter=1000):
    """Update coefficient first, then b using the NEW coefficient.

    Each history row is one complete sweep, including the initial parameters.
    Convergence requires both MSE partial derivatives to be small.
    """
    z, y = _data(points, power)
    if not np.isfinite(tol) or tol <= 0 or not isinstance(max_iter, int) or max_iter < 0:
        raise ValueError('Require positive finite tol and nonnegative integer max_iter')
    coefficient, b = map(float, initial)
    if not np.all(np.isfinite([coefficient, b])):
        raise ValueError('Initial parameters must be finite')
    h_coefficient = 2*float(np.mean(z*z))
    history = []
    converged = False
    for iteration in range(max_iter + 1):
        residual = coefficient*z+b-y
        mse = float(np.mean(residual**2))
        gradient_c = 2*float(np.mean(residual*z))
        gradient_b = 2*float(np.mean(residual))
        if not np.all(np.isfinite([mse, gradient_c, gradient_b])):
            raise ValueError('Nonfinite fit evaluation')
        history.append(dict(iteration=iteration, coefficient=coefficient, b=b, mse=mse))
        if max(abs(gradient_c), abs(gradient_b)) <= tol:
            converged = True
            break
        if iteration == max_iter:
            break
        # For these data: Hessian diagonals are (7,2) or (49,2).
        coefficient -= gradient_c/h_coefficient
        gradient_b_new = 2*float(np.mean(coefficient*z+b-y))
        b -= gradient_b_new/2
    return dict(coefficient=coefficient, b=b, mse=mse, history=history,
                converged=converged, status='converged' if converged else 'max_iter')


def plot_fit(points, power, result, exact, path):
    import matplotlib.pyplot as plt
    x, y = np.asarray(points).T
    xs = np.linspace(x.min()-0.3, x.max()+0.3, 400)
    fig, (ax, convergence) = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    history = result['history']
    for i in (0, 1, 2, 3, 6):
        if i < len(history)-1:
            h = history[i]
            ax.plot(xs, h['coefficient']*xs**power+h['b'], '--', alpha=0.6,
                    linewidth=1, label=f'Iteration {i}')
    ax.scatter(x, y, color='black', s=45, label='Data', zorder=5)
    ax.plot(xs, result['coefficient']*xs**power+result['b'], color='crimson',
            linewidth=2, label='Final Newton fit')
    model = 'mx + b' if power == 1 else 'ax² + b'
    ax.set(xlabel='x', ylabel='y', title=f'y = {model}; MSE = {result["mse"]:.7f}')
    ax.legend(fontsize=8)
    convergence.semilogy([h['iteration'] for h in history],
                        [max(abs(h['mse']-exact['mse']), 1e-16) for h in history], color='navy')
    convergence.set(xlabel='Complete Newton sweep', ylabel='|MSE − analytical MSE|',
                    title='Convergence (display floor: 10⁻¹⁶)')
    for axis in (ax, convergence):
        axis.grid(alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    output = Path(__file__).parent / 'plots'
    output.mkdir(exist_ok=True)
    results = {}
    for name, power in [('line', 1), ('parabola', 2)]:
        exact = analytical_fit(power=power)
        numerical = sequential_newton_fit(power=power)
        assert numerical['converged']
        for key in ('coefficient', 'b', 'mse'):
            assert abs(exact[key]-numerical[key]) < 1e-8
        print(f'{name}: analytical={exact}; Newton sweeps={len(numerical["history"])-1}')
        results[name] = dict(analytical=exact, newton=numerical)
        plot_fit(POINTS, power, numerical, exact, output / f'part2_{name}.png')
    (output / 'part2_history.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
