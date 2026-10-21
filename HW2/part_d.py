"""Part D: select degree-8 polynomial ridge regularization by 10-fold CV."""

from part_a import ROOT, polynomial_basis, rmse
import matplotlib.pyplot as plt
import numpy as np

LAMBDAS = np.array([0, 0.01, 0.1, 1, 10, 100, 1000])


def polynomial_regression_reg(x_train, y_train, regularization, degree=8):
    """Minimize sum of squared errors + lambda * sum(non-bias weights**2).

    Standardize the input using this training subset only, then expand its
    powers. The intercept is not penalized. Augmented least squares avoids
    forming/inverting the potentially ill-conditioned normal equations.
    """
    if regularization < 0:
        raise ValueError("regularization must be nonnegative")
    # website-excerpt-start
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    basis = polynomial_basis((x_train - mean) / scale, degree)
    penalty = np.eye(basis.shape[1])[1:]
    augmented = np.vstack([basis, np.sqrt(regularization) * penalty])
    targets = np.concatenate([y_train, np.zeros(len(penalty))])
    weights = np.linalg.lstsq(augmented, targets, rcond=None)[0]
    # website-excerpt-end
    return {"weights": weights, "mean": mean, "scale": scale, "degree": degree}


def predict(model, x):
    return polynomial_basis((x - model["mean"]) / model["scale"],
                            model["degree"]) @ model["weights"]


def cross_validate(x, y, lambdas=LAMBDAS, seed=572):
    """Use identical reproducible shuffled folds for every lambda.

    Only pass the first 100 observations; the held-out test set is not used.
    Each fold learns its normalization on its own 90 training observations.
    """
    if len(x) != 100 or len(y) != 100:
        raise ValueError("Part D cross-validation requires the first 100 rows")
    # website-excerpt-start
    folds = np.array_split(np.random.default_rng(seed).permutation(len(y)), 10)
    errors = np.empty((len(lambdas), 10))
    for fold_index, validation in enumerate(folds):
        training = np.concatenate([fold for i, fold in enumerate(folds) if i != fold_index])
        for j, regularization in enumerate(lambdas):
            model = polynomial_regression_reg(x[training], y[training], regularization)
            errors[j, fold_index] = rmse(y[validation], predict(model, x[validation]))
    return errors
    # website-excerpt-end


def main():
    data = np.loadtxt(ROOT / "data/boston_house_prices.csv", delimiter=",", skiprows=2)
    x, y = data[:100, 1:2], data[:100, -1]
    # website-excerpt-start
    errors = cross_validate(x, y)
    means = errors.mean(axis=1)
    best = int(np.argmin(means))
    chosen = LAMBDAS[best]
    # website-excerpt-end
    plots, output = ROOT / "plots", ROOT / "results"
    plots.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)
    np.savetxt(output / "part_d_cv_errors.csv",
               np.column_stack([LAMBDAS, means, errors]), delimiter=",",
               header="lambda,mean_validation_rmse," + ",".join(f"fold_{i}_rmse" for i in range(1, 11)),
               comments="", fmt="%.10g")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(LAMBDAS[1:], means[1:], "o-", label="Mean 10-fold validation RMSE")
    ax.axhline(means[0], color="tab:orange", linestyle="--",
               label=f"Unregularized baseline (lambda = 0): {means[0]:.3f}")
    if chosen > 0:
        ax.plot(chosen, means[best], "*", color="tab:red", markersize=15,
                label=f"Selected lambda = {chosen:g}")
    ax.set(xlabel="Regularization constant lambda (logarithmic scale)",
           ylabel="Mean validation RMSE ($1000)",
           title="Part D: degree-8 polynomial, second feature (ZN)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "part_d_cross_validation.png", dpi=200)
    plt.close(fig)

    (output / "part_d_notes.md").write_text(f"""# Part D: regularized polynomial regression

Based on the 10-fold cross-validation results, choosing **λ = {chosen:g}**
gives the lowest mean validation RMSE of **{means[best]:.6f}** ($1000) among
the tested values. The unregularized baseline (λ = 0) has a mean validation
RMSE of {means[0]:.6f}, shown as a horizontal line.
""")
    # Preserve the selected model refitted on all 100 training observations.
    model = polynomial_regression_reg(x, y, chosen)
    np.savez(output / "part_d_model.npz", **model, regularization=chosen)
    for value, error in zip(LAMBDAS, means):
        print(f"lambda = {value:7g}: mean validation RMSE = {error:.6f}")
    print(f"Selected lambda: {chosen:g}")


if __name__ == "__main__":
    main()
