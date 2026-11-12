# Assignment 4 - Naive Bayes sentiment classification

The first implementation covers all required tasks and the optional learning
curve. See the [assignment website](https://ayaaniqbal.github.io/SYDE572/HW4/) and [report](report.md). Held-out accuracy is **81.54%**
(1,943 / 2,383). The classifier uses no machine-learning toolkit.

From the repository root:

```sh
.venv/bin/python -m pip install -r HW4/requirements.txt
MPLCONFIGDIR=.mplconfig .venv/bin/python HW4/run_assignment.py
.venv/bin/python -m unittest discover -s HW4 -p 'test_*.py'
```

The runner resolves paths relative to itself and can also run from elsewhere.
It regenerates metrics, predictions, error examples, learning-curve data, and
plots. The written report is maintained separately.

- `data/all_sentiment_shuffled.txt`: original review corpus from the supplied ZIP.
- `naive_bayes.py`: reusable classifier and required assignment functions.
- `run_assignment.py`: evaluation, sanity checks, error examples, and plots.
- `test_naive_bayes.py`: independent analytical checks.
- `results/`: numerical results and traceable predictions.
- `plots/evaluation.png`: learning curve and confusion matrix.
