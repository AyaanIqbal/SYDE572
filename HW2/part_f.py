"""Part F: 90 Gaussian basis functions with L2 regularization and 10-fold CV."""

from part_e import ROOT, gaussian_basis, rmse
from part_d import LAMBDAS
import matplotlib.pyplot as plt
import numpy as np


def gaussian_regression_reg(x_train, y_train, regularization, n_basis=90,
                            sigma=1.0, seed=572):
    """Fit SSE + lambda * squared non-bias weights using training-only centres."""
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative")
    if not 1 <= n_basis <= len(x_train):
        raise ValueError("n_basis must not exceed the training size")
    # website-excerpt-start
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    train = (x_train - mean) / scale
    indices = np.random.default_rng(seed).permutation(len(train))[:n_basis]
    centres = train[indices]
    basis = gaussian_basis(train, centres, sigma)
    penalty = np.eye(n_basis + 1)[1:]
    augmented = np.vstack([basis, np.sqrt(regularization) * penalty])
    targets = np.concatenate([y_train, np.zeros(n_basis)])
    weights = np.linalg.lstsq(augmented, targets, rcond=None)[0]
    # website-excerpt-end
    return {"weights": weights, "mean": mean, "scale": scale,
            "centres": centres, "centre_indices": indices, "sigma": sigma,
            "regularization": regularization}


def predict(model, x):
    return gaussian_basis((x - model["mean"]) / model["scale"],
                          model["centres"], model["sigma"]) @ model["weights"]


def cross_validate(x, y, lambdas=LAMBDAS, seed=572):
    """Reuse identical folds/centres across lambdas; exclude validation centres.

    Each fold has 90 training rows, so all 90 become Gaussian centres.
    Both scaling and centre selection are refitted inside each fold.
    """
    if len(x) != 100 or len(y) != 100:
        raise ValueError("Part F requires the first 100 observations")
    # website-excerpt-start
    folds = np.array_split(np.random.default_rng(seed).permutation(100), 10)
    errors = np.empty((len(lambdas), 10))
    for i, validation in enumerate(folds):
        training = np.concatenate([fold for j, fold in enumerate(folds) if j != i])
        for j, value in enumerate(lambdas):
            model = gaussian_regression_reg(x[training], y[training], value, seed=seed)
            errors[j, i] = rmse(y[validation], predict(model, x[validation]))
    return errors
    # website-excerpt-end


def main():
    data = np.loadtxt(ROOT / "data/boston_house_prices.csv", delimiter=",", skiprows=2)
    x, y = data[:100, :-1], data[:100, -1]
    # website-excerpt-start
    errors = cross_validate(x, y)
    means = errors.mean(axis=1)
    best = int(np.argmin(means))
    chosen = LAMBDAS[best]
    # website-excerpt-end
    output, plots = ROOT / "results", ROOT / "plots"
    output.mkdir(exist_ok=True)
    plots.mkdir(exist_ok=True)
    np.savetxt(output / "part_f_cv_errors.csv", np.column_stack([LAMBDAS, means, errors]),
               delimiter=",", comments="", fmt="%.10g",
               header="lambda,mean_validation_rmse," + ",".join(f"fold_{i}_rmse" for i in range(1, 11)))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(LAMBDAS[1:], means[1:], "o-", label="Mean 10-fold validation RMSE")
    ax.axhline(means[0], color="tab:orange", linestyle="--",
               label=f"Unregularized baseline (lambda = 0): {means[0]:.3f}")
    if chosen > 0:
        ax.plot(chosen, means[best], "*", color="tab:red", markersize=15,
                label=f"Selected lambda = {chosen:g}")
    ax.set(xlabel="Regularization constant lambda (logarithmic scale)",
           ylabel="Mean validation RMSE ($1000)",
           title="Part F: 90 Gaussian basis functions (sigma = 1)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "part_f_cross_validation.png", dpi=200)
    plt.close(fig)
    np.savez(output / "part_f_model.npz", **gaussian_regression_reg(x, y, chosen))
    (output / "part_f_notes.md").write_text(f"""# Part F: regularized Gaussian regression

Based on the 10-fold cross-validation results, choosing **λ = {chosen:g}**
gives the lowest mean validation RMSE of **{means[best]:.6f}** ($1000) among
the tested values. The unregularized baseline (λ = 0) has a mean validation
RMSE of {means[0]:.6f}, shown as a horizontal line.
""")
    for value, error in zip(LAMBDAS, means):
        print(f"lambda = {value:7g}: mean validation RMSE = {error:.6f}")
    print(f"Selected lambda: {chosen:g}")


if __name__ == "__main__":
    main()
