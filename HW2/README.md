# Assignment 2 — Part A

From the repository root:

```sh
python -m pip install -r HW2/requirements.txt
python HW2/part_a.py
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
