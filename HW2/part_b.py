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

Coefficients predict house value in $1000. Normalized weights correspond to
one training standard deviation; original-unit weights correspond to one
unit of the original feature. The bias is included below.

{chr(10).join(table)}

**RM (6th feature):** One additional average room predicts about
${slopes[rm] * 1000:,.0f} higher value, holding other features fixed. The
positive coefficient is reasonable because larger homes generally cost more.

**DIS (8th feature):** One additional distance unit predicts about
${slopes[dis] * 1000:,.2f} higher value, holding other features fixed. The
positive sign is less intuitive if proximity to employment increases value,
but the effect is tiny and may reflect correlated features and the limited
training sample. These associations are not causal conclusions.

CHAS is constant in the training data, so its zero coefficient is not evidence
of no effect; its influence cannot be estimated from this subset.
"""
    (output / "part_b_notes.md").write_text(report)
    print("\n".join(table))
    print(f"\nSaved weights and report notes to {output}")


if __name__ == "__main__":
    main()
