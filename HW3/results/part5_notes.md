For ISOMAP, each flower was linked to its three closest neighbours, and the resulting graph distances were represented in a 2D plot. These paths can follow curved structure in the measurements.

Setosa stands apart in both plots, but the boundary between versicolor and virginica remains mixed. The curved ISOMAP layout does not make that boundary noticeably easier to distinguish.

For compression, PCA is preferred since its first two components retain 97.77% of the variance and allow direct reconstruction of the original measurements. ISOMAP aims to preserve graph distances rather than minimize reconstruction error.

The three-neighbour graph also has two disconnected groups, which scikit-learn connects automatically before producing the ISOMAP plot. This makes their relative placement dependent on added connections and weakens the case for using ISOMAP here. PCA is therefore the more suitable choice for this dataset based on compression and the displayed class structure, although the plots alone do not establish classification accuracy.
