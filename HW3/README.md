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

Parts 3–5 import plotting helpers from `part2.py`, so keep these files together.
You can run the scripts from any working directory, and they will save outputs
beside the source files. Once dependencies are installed, the scripts use the
bundled Iris data without needing network access.

PCA centers the measurements in centimetres without standardizing or whitening
them, and species labels are used only to colour the plots. All results use the
scikit-learn dataset, which includes two corrected observations compared with
the older UCI `iris.data` file.

## Files

- `handwritten/part1.pdf`: original handwritten Part 1 solution.
- `part2.py` and `plots/part2_features.png`: 1D, 2D, and 3D feature projections.
- `part3.py`, `plots/part3_pca.png`, and `results/pca_axes.csv`: PCA projections and axes.
- `part4.py`, `plots/part4_variance.png`, `results/explained_variance.csv`, and
  `results/part4_compression.json`: variance and reconstruction results.
- `part5.py`, `plots/part5_pca_isomap.png`, `results/isomap_embedding.csv`, and
  `results/part5_diagnostics.json`: ISOMAP comparison and graph diagnostics.
- `results/part2_notes.md` through `results/part5_notes.md`: written discussion.

The website loads notes, tables, and complete source directly from their original
files, while the HTML contains fallback discussion for readers without JavaScript.
If you edit the discussion, update both the notes and that fallback text to keep
them consistent. Running the Python scripts will not overwrite your notes.
In the 1D plots, the species rows are display offsets that make points easier to see.

## Preview and publication

Run `python3 -m http.server 8000` from the repository root and open
`http://localhost:8000/HW3/`. Use HTTP so tables and source can load.
The existing GitHub Pages setup publishes the repository root from `main`.
Local edits are not public until committed and pushed.

## Interpretation

Two PCs explain 97.7685% of the original variance, and PC1 alone exceeds the
50% target. The three-neighbour ISOMAP graph has two disconnected components,
which scikit-learn connects before producing the embedding. The diagnostics
preserve the resulting warnings so this limitation remains visible.

Scikit-learn uses classical MDS through kernel eigendecomposition for the embedding
step. See the website for the full comparison and source references.
