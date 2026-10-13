"""Assignment 2, Part A: polynomial basis regression without regularization."""

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT.parent / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def polynomial_basis(x, degree):
    """Return [1, x1, x1**2, ..., x2, x2**2, ...], without cross-terms."""
    if degree < 1:
        raise ValueError("degree must be at least 1")
    columns = [np.ones(x.shape[0])]
    columns.extend(x[:, j] ** power for j in range(x.shape[1])
                   for power in range(1, degree + 1))
    return np.column_stack(columns)


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def polynomial_regression(x_train, y_train, x_test, y_test, degree):
    """Normalize on training data and fit unregularized least squares.

    The target remains in its original units ($1000). Singular-value-based
    least squares handles redundant columns (e.g. powers of binary features)
    without explicitly inverting the normal-equation matrix.
    """
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    train_basis = polynomial_basis((x_train - mean) / scale, degree)
    test_basis = polynomial_basis((x_test - mean) / scale, degree)
    weights, _, rank, _ = np.linalg.lstsq(train_basis, y_train, rcond=None)
    return {
        "degree": degree,
        "weights": weights,
        "mean": mean,
        "scale": scale,
        "rank": int(rank),
        "train_rmse": rmse(y_train, train_basis @ weights),
        "test_rmse": rmse(y_test, test_basis @ weights),
    }


def main():
    # The first two CSV lines contain dataset metadata and column names.
    data = np.loadtxt(ROOT / "data/boston_house_prices.csv", delimiter=",", skiprows=2)
    if data.shape != (506, 14) or not np.isfinite(data).all():
        raise ValueError("Expected 506 finite rows with 13 features and one target")
    x, y = data[:, :-1], data[:, -1]
    results = [polynomial_regression(x[:100], y[:100], x[100:], y[100:], degree)
               for degree in range(1, 8)]

    output = ROOT / "plots"
    output.mkdir(exist_ok=True)
    rows = np.array([[r["degree"], r["train_rmse"], r["test_rmse"], r["rank"]]
                     for r in results])
    np.savetxt(output / "part_a_errors.csv", rows, delimiter=",",
               header="degree,train_rmse,test_rmse,design_rank", comments="",
               fmt=["%d", "%.10g", "%.10g", "%d"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rows[:, 0], rows[:, 1], "o-", label="Training (first 100 rows)")
    ax.plot(rows[:, 0], rows[:, 2], "s-", label="Test (remaining 406 rows)")
    # High-degree extrapolation can make test error orders of magnitude larger.
    # A log y-axis keeps both curves visible, while still showing actual RMSE.
    ax.set_yscale("log")
    ax.set(xlabel="Polynomial degree", ylabel="RMSE ($1000; logarithmic scale)",
           title="Part A: unregularized polynomial regression", xticks=range(1, 8))
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "part_a_rmse.png", dpi=200)
    plt.close(fig)

    print("Degree    Training RMSE    Test RMSE    Rank")
    for degree, train, test, rank in rows:
        print(f"{int(degree):6d}    {train:13.6f}    {test:9.6f}    {int(rank):4d}")
    print(f"\nSaved plot and errors to {output}")


if __name__ == "__main__":
    main()
