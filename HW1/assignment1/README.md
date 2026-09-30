# Homework 1 Python

Run from the repository root:

```sh
python -m pip install -r HW1/requirements.txt
python HW1/assignment1/part1.py
python HW1/assignment1/part2.py
python -m unittest discover -s HW1/assignment1 -v
```

`part1.py` implements Newton–Raphson and Golden Section closest-point searches.
Examples use x² + 5, x³ − 2x + 1, and exp(x). Newton uses a derivative tolerance
of 1e-10; Golden Section uses an interval width of 1e-8. Both accept iteration limits.

`part2.py` fits y = mx + b and y = ax² + bx + c. The analytical method solves the
normal equations. The numerical method updates each coefficient in order, using
the newest values. Running the script shows three iterations from zero, matching
the handwritten procedure. Call `sequential_newton_fit` with a larger `max_iter`
to continue; its default limit is 1000 and derivative tolerance is 1e-10.

The analytical line is y = 2.3x − 0.2 (MSE 0.575). The analytical parabola is
 y = 0.75x² + 0.05x + 0.55 (MSE 0.0125). After three sequential iterations,
the parabola MSE is approximately 0.023331829.

Plots are saved in `plots/`. Tests compare against independent least-squares
solutions, check the first three parabola iterations, and exercise search safeguards.
