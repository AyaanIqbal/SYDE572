"""Part E: unregularized Gaussian basis regression with training centres."""

from part_a import ROOT, rmse
import matplotlib.pyplot as plt
import numpy as np


def gaussian_basis(x, centres, sigma=1.0):
    """Bias plus exp(-||x-centre||^2 / (2 sigma^2)); inputs standardized."""
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite")
    squared_distances = np.sum((x[:, None, :] - centres[None, :, :]) ** 2, axis=2)
    return np.column_stack([np.ones(len(x)),
                            np.exp(-squared_distances / (2 * sigma ** 2))])


def gaussian_regression(x_train, y_train, x_test, y_test, n_basis,
                        sigma=1.0, seed=572):
    """Fit K Gaussian functions plus a bias, without regularization.

    Reusing the seed yields nested sets of centres for different K values.
    Only training observations determine normalization and centre selection.
    """
    if not 1 <= n_basis <= len(x_train):
        raise ValueError("n_basis must be between 1 and the training size")
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    train = (x_train - mean) / scale
    test = (x_test - mean) / scale
    indices = np.random.default_rng(seed).permutation(len(train))[:n_basis]
    centres = train[indices]
    train_basis = gaussian_basis(train, centres, sigma)
    test_basis = gaussian_basis(test, centres, sigma)
    weights, _, rank, _ = np.linalg.lstsq(train_basis, y_train, rcond=None)
    return {"n_basis": n_basis, "weights": weights, "mean": mean,
            "scale": scale, "centres": centres, "centre_indices": indices,
            "sigma": sigma, "rank": int(rank),
            "train_rmse": rmse(y_train, train_basis @ weights),
            "test_rmse": rmse(y_test, test_basis @ weights)}


def main():
    data = np.loadtxt(ROOT / "data/boston_house_prices.csv", delimiter=",", skiprows=2)
    if data.shape != (506, 14) or not np.isfinite(data).all():
        raise ValueError("Expected 506 finite observations with 13 features and a target")
    x, y = data[:, :-1], data[:, -1]
    results = [gaussian_regression(x[:100], y[:100], x[100:], y[100:], k)
               for k in range(5, 96, 10)]
    rows = np.array([[r["n_basis"], r["train_rmse"], r["test_rmse"], r["rank"]]
                     for r in results])
    plots, output = ROOT / "plots", ROOT / "results"
    plots.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)
    np.savetxt(output / "part_e_errors.csv", rows, delimiter=",",
               header="n_basis,train_rmse,test_rmse,design_rank", comments="",
               fmt=["%d", "%.10g", "%.10g", "%d"])
    np.savetxt(output / "part_e_centre_order.csv",
               np.random.default_rng(572).permutation(100), fmt="%d",
               header="training_row_index_zero_based", comments="")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rows[:, 0], rows[:, 1], "o-", label="Training (first 100 rows)")
    ax.plot(rows[:, 0], rows[:, 2], "s-", label="Test (remaining 406 rows)")
    ax.set(xlabel="Number of Gaussian basis functions (excluding bias)",
           ylabel="RMSE ($1000)", xticks=range(5, 96, 10),
           title="Part E: unregularized Gaussian regression (sigma = 1)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "part_e_rmse.png", dpi=200)
    plt.close(fig)
    table = "\n".join(f"| {int(k)} | {train:.6f} | {test:.6f} |"
                       for k, train, test, _ in rows)
    (output / "part_e_notes.md").write_text(f"""# Part E: Gaussian basis regression

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
{table}

Training RMSE falls from {rows[0, 1]:.3f} at K = 5 to {rows[-1, 1]:.3f}
at K = 95 as additional basis functions increase model flexibility. Test
RMSE stays around 9.8 through K = 65, then jumps to about 11.9 at K = 75
and remains above 11.7. The widening gap is evidence of overfitting.
Test error does not have to decrease with training error: the centres cover only
the first 100 observations, and a flexible model may generalize poorly to
the other observations. Gaussian functions are local, so predictions far
from every centre approach the fitted bias. Interpret the plotted test
curve as held-out evaluation, rather than using it to tune the width or
centre selection. Part F will use cross-validation for regularization.
""")
    for k, train, test, rank in rows:
        print(f"K = {int(k):2d}: training RMSE = {train:.6f}, test RMSE = {test:.6f}, rank = {int(rank)}")


if __name__ == "__main__":
    main()
