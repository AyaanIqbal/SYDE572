# Assignment 2 — Parts A–C

From the repository root:

```sh
python -m pip install -r HW2/requirements.txt
python HW2/part_a.py
python HW2/part_b.py
python HW2/part_c.py
```

With this repository's existing environment, run `.venv/bin/python HW2/part_a.py`.

The script fits unregularized polynomial regression for degrees 1–7. It uses
the first 100 observations for training and the remaining 406 for testing,
without shuffling. All 13 features are standardized with training-only means
and standard deviations. Targets retain their original units of $1000.
The design matrix includes a bias and individual feature powers, with no
cross-terms. `numpy.linalg.lstsq` solves least squares without regularization;
it handles redundant columns using its standard numerical rank tolerance.

Outputs:

- `plots/part_a_rmse.png`: training and test RMSE versus polynomial degree.
  The RMSE axis is logarithmic so both curves remain visible when high-degree
  test errors grow very large.
- `plots/part_a_errors.csv`: exact errors and numerical design-matrix ranks.

The local CSV preserves the original Boston Housing observation and feature
order. Source: [scikit-learn 1.1.3 dataset CSV](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.1.3/sklearn/datasets/data/boston_house_prices.csv).
Dataset description: [the assignment's Toronto reference](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html).

## Part B

Status: draft only; pending review.

`part_b.py` reuses Part A's degree-1 fit, prints all weights (including the
bias), and converts them to original feature units. It verifies that the
converted coefficients preserve predictions on every observation.
It saves `results/part_b_weights.csv` and report-ready discussion in
`results/part_b_notes.md`, including interpretation of RM and DIS and the
constant CHAS feature in the training subset.

## Part C

Run `.venv/bin/python HW2/part_c.py` to fit degrees 1, 3, and 7 using only ZN,
the second feature (Python index 1). The first 100 rows remain the training
set. The same training-only normalization is applied to test points and the
dense plotting grid; axes show original feature and target units.

Each `plots/part_c_degree_*.png` shows training points, test points, and the
learned polynomial across the complete observed feature range. Shading marks
values beyond the training feature range. Errors are saved to
`results/part_c_errors.csv`; brief report comments are in
`results/part_c_notes.md`.
