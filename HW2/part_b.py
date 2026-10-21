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
    print("\n".join(table))
    print(f"\nSaved weights to {output}")


if __name__ == "__main__":
    main()
