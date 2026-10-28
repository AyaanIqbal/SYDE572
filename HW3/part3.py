"""Reproduce Assignment 3, Part 3. Run from any working directory."""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

from part2 import ROOT, projections


def main():
    (ROOT / 'plots').mkdir(exist_ok=True)
    (ROOT / 'results').mkdir(exist_ok=True)
    iris = load_iris()

    # PCA centers the original centimetre measurements internally.
    # Do not standardize or whiten: retain variance in the original units.
    pca = PCA(n_components=4, svd_solver='full')
    scores = pca.fit_transform(iris.data)
    np.testing.assert_allclose(
        pca.components_ @ pca.components_.T, np.eye(4), atol=1e-12)

    projections(scores, iris.target, iris.target_names,
                ['PC1 (cm)', 'PC2 (cm)', 'PC3 (cm)'],
                'part3_pca.png', 'Part 3: Iris principal-component projections')
    # Each row is a unit PCA axis in the original four-feature space.
    np.savetxt(ROOT / 'results/pca_axes.csv', pca.components_, delimiter=',',
               header=','.join(iris.feature_names), comments='')

    print('Feature order:', ', '.join(iris.feature_names))
    print('Centering mean:', pca.mean_)
    for index, axis in enumerate(pca.components_, start=1):
        print(f'PC{index}: {axis}')


if __name__ == '__main__':
    main()
