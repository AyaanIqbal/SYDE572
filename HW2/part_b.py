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
    # website-excerpt-start
    result = polynomial_regression(x[:100], y[:100], x[100:], y[100:], 1)
    weights = result["weights"]
    slopes = weights[1:] / result["scale"]
    intercept = weights[0] - result["mean"] @ slopes
    # website-excerpt-end

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
unit of the original feature.

{chr(10).join(table)}

## Key coefficients

| Variable | Normalized-model coefficient | Original-unit coefficient |
| --- | ---: | ---: |
| Bias | {weights[0]:.6f} | {intercept:.6f} |
| Rooms (RM, 6th feature) | {weights[rm + 1]:.6f} | {slopes[rm]:.6f} |
| Distance (DIS, 8th feature) | {weights[dis + 1]:.6f} | {slopes[dis]:.6f} |

**Bias:** The normalized-model bias predicts ${weights[0] * 1000:,.0f} when
all features equal their training means. This is a reasonable baseline: it
equals the average training house value because the inputs are centered.

**Rooms (RM): Reasonable.** Holding other features fixed, one additional
average room predicts about ${slopes[rm] * 1000:,.0f} higher value. The positive
coefficient agrees with the expectation that larger homes generally cost more.

**Distance (DIS): The positive sign is contrary to the expected relationship.**
Holding other features fixed, one additional distance unit predicts only
${slopes[dis] * 1000:,.2f} higher value. If proximity to employment increases
value, a negative coefficient would be more intuitive. However, the effect
is nearly zero, so this is weak evidence of a positive relationship; correlated
features and the limited training sample may explain the unexpected sign.

CHAS is constant in the training data, so its zero coefficient is not
evidence of no effect.
"""
    (output / "part_b_notes.md").write_text(report)
    print("\n".join(table))
    print(f"\nSaved weights and report notes to {output}")


if __name__ == "__main__":
    main()
