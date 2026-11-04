ISOMAP uses three nearest neighbours and a two-dimensional embedding of the same four measurements. It estimates geodesic distances through shortest paths on the neighbourhood graph. Scikit-learn uses a kernel eigendecomposition for the classical MDS embedding step, rather than the gradient-descent MDS described in the handout.

The original undirected graph has two connected components: 50 setosa flowers and 100 versicolor/virginica flowers. Scikit-learn adds cross-component connections to produce the embedding. The relative placement of the groups therefore depends on added connectivity; the saved diagnostics retain the warnings.

Both plots isolate setosa. ISOMAP spreads the other species along a curved structure, but they still overlap near the transition. I prefer PCA for compression here: two PCs retain 97.77% of variance, permit direct reconstruction, and minimize squared reconstruction error among rank-two linear representations. ISOMAP targets graph geometry and has no directly comparable PCA explained-variance percentage or built-in inverse transform.

PCA is also a reasonable visualization default for this dataset, given the disconnected three-neighbour graph. These plots do not establish which method gives higher classification accuracy; that would require held-out evaluation with dimensionality reduction fitted inside each training fold.
