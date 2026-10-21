"""Extract and interpret the degree-1 weights from Part A."""

import csv
import numpy as np
from part_a import ROOT, polynomial_basis, polynomial_regression


def main():
    path = ROOT / "data/boston_house_prices.csv"
    with path.open(newline="") as source:
        reader = csv.reader(source)
        next(reader)
        names = next(reader)[:-1]
    data = np.loadtxt(path, delimiter=",", skiprows=2)
    x, y = data[:, :-1], data[:, -1]
    result = polynomial_regression(x[:100], y[:100], x[100:], y[100:], 1)
    weights = result["weights"]
    slopes = weights[1:] / result["scale"]
    intercept = weights[0] - result["mean"] @ slopes

    # Converting units must preserve predictions on both train and test rows.
    normalized = (x - result["mean"]) / result["scale"]
    np.testing.assert_allclose(polynomial_basis(normalized, 1) @ weights,
                               intercept + x @ slopes, rtol=1e-10, atol=1e-10)

    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    with (output / "part_b_weights.csv").open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["feature", "normalized_weight", "original_unit_weight"])
        writer.writerow(["bias", weights[0], intercept])
        writer.writerows(zip(names, weights[1:], slopes))

    table = ["| Feature | Normalized weight | Original-unit weight |",
             "| --- | ---: | ---: |",
             f"| Bias | {weights[0]:.6f} | {intercept:.6f} |"]
    table.extend(f"| {name} | {weight:.6f} | {slope:.6f} |"
                 for name, weight, slope in zip(names, weights[1:], slopes))
    rm, dis = names.index("RM"), names.index("DIS")
    report = f"""# Part B: degree-1 weights

The model uses the first 100 observations, all 13 features, and a bias.
Features are standardized as z_j = (x_j - mean_j) / std_j, using training
statistics. The target is not normalized and is measured in $1000.

{chr(10).join(table)}

The normalized model is y_hat = w_0 + sum(w_j z_j). Each normalized feature
weight describes the predicted change in $1000 for a one-training-standard-
deviation increase, holding other features fixed. To interpret changes in
original units, use beta_j = w_j / std_j and
beta_0 = w_0 - sum(beta_j mean_j). These two forms give identical predictions.

## Rooms per dwelling (RM)

The normalized weight is {weights[rm + 1]:.6f}. The training standard deviation
is {result['scale'][rm]:.6f} rooms, so the original-unit coefficient is
{slopes[rm]:.6f} ($1000 per room). Holding all other features fixed, one more
room per dwelling increases the predicted median house value by approximately
${slopes[rm] * 1000:,.0f}. The positive sign is reasonable: larger homes would
generally be expected to have higher values. This is a fitted association,
not a causal estimate.

## Distance to employment centres (DIS)

The normalized weight is {weights[dis + 1]:.6f}. The training standard deviation
is {result['scale'][dis]:.6f} distance units, so the original-unit coefficient
is {slopes[dis]:.6f} ($1000 per dataset distance unit). Holding other features
fixed, a one-unit increase in weighted distance increases predicted value by
approximately ${slopes[dis] * 1000:,.2f}. This effect is very small compared
with RM: a one-standard-deviation increase changes the prediction by only
${weights[dis + 1] * 1000:,.2f}. Its positive sign may initially seem
surprising if proximity to employment is expected to increase value. However,
this coefficient describes a conditional association in the first 100 rows;
correlated features and the restricted training sample can affect its sign.
The fit alone does not establish why this association occurs.

## Indexing and limitations

RM and DIS are Python feature indices 5 and 7 (CSV feature positions 6 and 8).
With the bias first, they are entries 6 and 8 in the weight vector. Identify
them by name. These are the 6th and 8th features specified in the instructor's
assignment correction; the corrected numbering does not change this fit.

CHAS is zero throughout the training subset. Its coefficient is therefore
unidentifiable from these observations; the least-squares solver returns zero
in its minimum-norm solution. This is not evidence that CHAS has no effect.
The model has training RMSE {result['train_rmse']:.6f} and test RMSE
{result['test_rmse']:.6f} (both in $1000), so interpretation of coefficient
magnitudes should account for its poor generalization to the test subset.
"""
    (output / "part_b_notes.md").write_text(report)
    print("\n".join(table))
    print(f"\nSaved weights and report notes to {output}")


if __name__ == "__main__":
    main()
