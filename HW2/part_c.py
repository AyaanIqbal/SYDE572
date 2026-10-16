"""Part C: visualize polynomial fits using only the second feature (ZN)."""

from part_a import ROOT, polynomial_basis, polynomial_regression
import matplotlib.pyplot as plt
import numpy as np


def polynomial_regression_1d_vis(x_train, y_train, x_test, y_test, degree):
    """Fit one feature, and plot its curve in original feature units.

    Inputs x_train and x_test must each have shape (n, 1). Normalization is
    learned exclusively from training data by the shared Part A fitter.
    Returns the fitted model and a figure; the caller saves/closes the figure.
    """
    if x_train.ndim != 2 or x_train.shape[1] != 1:
        raise ValueError("x_train must contain exactly one feature column")
    if x_test.ndim != 2 or x_test.shape[1] != 1:
        raise ValueError("x_test must contain exactly one feature column")
    result = polynomial_regression(x_train, y_train, x_test, y_test, degree)
    grid = np.linspace(min(x_train.min(), x_test.min()),
                       max(x_train.max(), x_test.max()), 1500).reshape(-1, 1)
    normalized_grid = (grid - result["mean"]) / result["scale"]
    predictions = polynomial_basis(normalized_grid, degree) @ result["weights"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(x_test[:, 0], y_test, s=22, alpha=0.45, color="tab:orange",
               label="Test (406 points)")
    ax.scatter(x_train[:, 0], y_train, s=30, alpha=0.75, color="tab:blue",
               label="Training (100 points)")
    ax.plot(grid[:, 0], predictions, color="black", linewidth=2,
            label=f"Degree {degree} polynomial")
    if x_test.max() > x_train.max():
        ax.axvspan(x_train.max(), grid.max(), color="grey", alpha=0.1,
                   label="Beyond training feature range")
    ax.set(xlabel="ZN: residential land zoned for lots over 25,000 sq.ft. (%)",
           ylabel="Median house value ($1000)",
           title=f"Part C: single-feature polynomial regression (degree {degree})")
    ax.text(0.02, 0.97,
            f"Training RMSE: {result['train_rmse']:.3f}   |   Test RMSE: {result['test_rmse']:.3f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    ax.margins(y=0.15)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return result, fig


def main():
    data = np.loadtxt(ROOT / "data/boston_house_prices.csv", delimiter=",", skiprows=2)
    # MATLAB X(:, 2) corresponds to Python column index 1, the ZN feature.
    x, y = data[:, 1:2], data[:, -1]
    plots, output = ROOT / "plots", ROOT / "results"
    plots.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)
    rows = []
    for degree in (1, 3, 7):
        result, fig = polynomial_regression_1d_vis(x[:100], y[:100], x[100:], y[100:], degree)
        fig.savefig(plots / f"part_c_degree_{degree}.png", dpi=200)
        plt.close(fig)
        rows.append([degree, result["train_rmse"], result["test_rmse"]])
    np.savetxt(output / "part_c_errors.csv", rows, delimiter=",",
               header="degree,train_rmse,test_rmse", comments="",
               fmt=["%d", "%.10g", "%.10g"])
    for degree, train, test in rows:
        print(f"Degree {degree}: training RMSE = {train:.6f}, test RMSE = {test:.6f}")


if __name__ == "__main__":
    main()
