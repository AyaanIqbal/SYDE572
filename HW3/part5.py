"""Reproduce Assignment 3, Part 5. Run from any working directory."""
import json
import warnings

from part2 import ROOT, scatter
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
from scipy.sparse.csgraph import connected_components
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neighbors import kneighbors_graph


def main():
    for folder in ['plots', 'results']:
        (ROOT / folder).mkdir(exist_ok=True)
    iris = load_iris()
    X, y, names = iris.data, iris.target, iris.target_names
    # Match Parts 3–4: original centimetre measurements, without scaling.
    pca_scores = PCA(n_components=2, svd_solver='full').fit_transform(X)

    # Inspect connectivity before Isomap automatically completes the graph.
    # directed=False uses the undirected union of neighbour connections.
    graph = kneighbors_graph(X, n_neighbors=3, mode='distance', include_self=False)
    count, membership = connected_components(graph, directed=False)
    model = Isomap(n_neighbors=3, n_components=2,
                   eigen_solver='dense', path_method='D')
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        embedding = model.fit_transform(X)
    if not np.isfinite(embedding).all():
        raise ValueError('ISOMAP returned non-finite coordinates')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    panels = [
        (axes[0], pca_scores, 'PCA: first two components', 'PC'),
        (axes[1], embedding,
         f'ISOMAP: 3 neighbours ({count} graph components)', 'ISOMAP'),
    ]
    for ax, coordinates, title, label in panels:
        scatter(ax, coordinates, y, names)
        ax.set(xlabel=f'{label} 1', ylabel=f'{label} 2', title=title)
        ax.legend()
    fig.suptitle('Part 5: Two-dimensional representations of Iris', fontsize=14)
    fig.savefig(ROOT / 'plots/part5_pca_isomap.png', dpi=180)
    plt.close(fig)

    np.savetxt(ROOT / 'results/isomap_embedding.csv',
               np.column_stack([embedding, y]), delimiter=',',
               header='isomap_1,isomap_2,class', comments='')
    summary = {
        'n_neighbors': 3,
        'n_components': 2,
        'original_graph_components': int(count),
        'original_component_sizes': np.bincount(membership).tolist(),
        'component_species_counts': [
            {str(name): int(np.sum((membership == component) & (y == label)))
             for label, name in enumerate(names)}
            for component in range(count)
        ],
        'warnings': sorted(set(str(w.message) for w in captured)),
        'versions': {'numpy': np.__version__, 'scipy': scipy.__version__,
                     'scikit-learn': sklearn.__version__},
    }
    (ROOT / 'results/part5_diagnostics.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
