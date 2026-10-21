# Assignment 2 — Parts A–F

## Website updates

The course homepage links to `HW2/index.html`. This page displays plots directly
from `plots/` and loads the current notes, CSV tables, and Python source from
their original paths. There are no copied website results to synchronize.
After changing a model, rerun its Python script to regenerate its results,
then commit and push the changed source and outputs to `main`. GitHub Pages
publishes those changes using the repository's existing Pages configuration.
Local uncommitted edits do not update the public website. To preview locally,
run `python -m http.server 8000` from the repository root and visit
`http://localhost:8000/HW2/` (fetching notes requires HTTP, not a file URL).

From the repository root:

```sh
python -m pip install -r HW2/requirements.txt
python HW2/part_a.py
python HW2/part_b.py
python HW2/part_c.py
python HW2/part_d.py
python HW2/part_e.py
python HW2/part_f.py
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

## Part D

Run `.venv/bin/python HW2/part_d.py` to compare lambda values 0, 0.01, 0.1,
1, 10, 100, and 1000 for a degree-8 ZN-only polynomial. Ten-fold CV uses
only the first 100 observations, with shuffled folds (seed 572) shared across
candidates and normalization fitted separately within each training fold.
The objective is summed squared error plus lambda times squared non-bias
weights; the bias is not penalized. The script saves the semilog plot to
`plots/part_d_cross_validation.png`, fold errors and notes to `results/`,
and the selected model refitted on all 100 rows to `results/part_d_model.npz`.

## Part E

Run `.venv/bin/python HW2/part_e.py` to fit unregularized Gaussian regression
with 5, 15, ..., 95 basis functions plus a bias, using all 13 features.
The first 100 rows determine normalization and centres; the other 406 rows
are used only for testing. A fixed permutation (seed 572) supplies nested
centre sets. The Gaussian width is fixed at sigma = 1 in standardized feature
space, since the assignment does not specify it. No test-based tuning is used.
The script saves `plots/part_e_rmse.png`, `results/part_e_errors.csv`,
`results/part_e_centre_order.csv` (zero-based training indices), and
`results/part_e_notes.md`.

## Part F

Run `.venv/bin/python HW2/part_f.py` for 90 Gaussian basis functions using
all features, sigma = 1, and the same seven lambda candidates as Part D.
Ten-fold CV uses only the first 100 rows with seed 572. Each fold learns
normalization and selects centres exclusively from its 90 training rows;
all 90 rows become centres. The bias is unpenalized, and the objective uses
summed squared residuals. Outputs include `plots/part_f_cross_validation.png`,
`results/part_f_cv_errors.csv`, `results/part_f_notes.md`, and the selected
model refitted on all 100 rows in `results/part_f_model.npz`.
