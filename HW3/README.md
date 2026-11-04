# Assignment 3 — PCA and dimensionality reduction

The [assignment website](index.html) includes the handwritten solution, discussion,
plots, numerical tables, and downloadable source for all five parts.

## Run

From the repository root:

```sh
.venv/bin/python -m pip install -r HW3/requirements.txt
.venv/bin/python HW3/part2.py
.venv/bin/python HW3/part3.py
.venv/bin/python HW3/part4.py
.venv/bin/python HW3/part5.py
```

Parts 3–5 import plotting helpers from `part2.py`; keep these files together.
The scripts work from any working directory and save outputs beside the source.
After dependencies are installed, they use the bundled Iris data without network
access. PCA centers measurements in centimetres without standardizing or whitening.
Species labels colour plots; they do not fit either dimensionality reduction.
The scikit-learn dataset includes two corrected observations compared with the
older UCI `iris.data` file; all results use this version consistently.

## Files

- `handwritten/part1.pdf`: original handwritten Part 1 solution.
- `part2.py` and `plots/part2_features.png`: 1D, 2D, and 3D feature projections.
- `part3.py`, `plots/part3_pca.png`, and `results/pca_axes.csv`: PCA projections and axes.
- `part4.py`, `plots/part4_variance.png`, `results/explained_variance.csv`, and
  `results/part4_compression.json`: variance and reconstruction results.
- `part5.py`, `plots/part5_pca_isomap.png`, `results/isomap_embedding.csv`, and
  `results/part5_diagnostics.json`: ISOMAP comparison and graph diagnostics.
- `results/part2_notes.md` through `results/part5_notes.md`: written discussion.

The website loads current notes, tables, and complete source from their original
files. The HTML also contains fallback discussion for readers without JavaScript;
update it alongside the notes if editing the discussion. Scripts do not overwrite
notes. The 1D plots use species rows only as display offsets.

## Preview and publication

Run `python3 -m http.server 8000` from the repository root and open
`http://localhost:8000/HW3/`. Use HTTP so tables and source can load.
The existing GitHub Pages setup publishes the repository root from `main`.
Local edits are not public until committed and pushed.

## Interpretation

Two PCs explain 97.7685% of original variance; PC1 alone exceeds 50%.
The three-neighbour ISOMAP graph has two components. Scikit-learn completes the
graph to embed it; warnings are preserved in the diagnostics. Scikit-learn uses
classical MDS via kernel eigendecomposition, rather than iterative gradient-descent
MDS. See the website for the full comparison and source references.
