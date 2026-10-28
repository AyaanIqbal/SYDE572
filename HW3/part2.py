"""Reproduce Assignment 3, Part 2. Run from any working directory."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT.parent / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

COLORS = ['#2878b5', '#dd8452', '#55a868']
MARKERS = ['o', '^', 's']
plt.rcParams.update({'figure.dpi': 130, 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.facecolor': 'white'})


def scatter(ax, coordinates, labels, names):
    for c, name in enumerate(names):
        values = coordinates[labels == c]
        ax.scatter(*values.T, label=name, color=COLORS[c], marker=MARKERS[c],
                   s=30, alpha=.8, edgecolors='white', linewidths=.35)


def projections(values, labels, names, axis_names, filename, title):
    fig = plt.figure(figsize=(15, 4.7), layout='constrained')
    ax = fig.add_subplot(131)
    # Separate class rows make coincident 1D observations visible; y is not a feature.
    scatter(ax, np.column_stack([values[:, 0], labels]), labels, names)
    ax.set(yticks=range(3), yticklabels=names, xlabel=axis_names[0],
           title='1D projection (class rows for visibility)')
    ax = fig.add_subplot(132)
    scatter(ax, values[:, :2], labels, names)
    ax.set(xlabel=axis_names[0], ylabel=axis_names[1], title='2D projection')
    ax = fig.add_subplot(133, projection='3d')
    scatter(ax, values[:, :3], labels, names)
    ax.set(xlabel=axis_names[0], ylabel=axis_names[1], zlabel=axis_names[2],
           title='3D projection')
    ax.view_init(elev=23, azim=130)
    ax.legend(fontsize=8, loc='upper left')
    fig.suptitle(title, fontsize=15)
    fig.savefig(ROOT / 'plots' / filename, dpi=180, bbox_inches='tight', pad_inches=.25)
    plt.close(fig)


def main():
    (ROOT / 'plots').mkdir(exist_ok=True)
    iris = load_iris()
    X, y, names = iris.data, iris.target, iris.target_names
    # Part 2: marginalize by selecting columns, without fitting a model.
    selected = [2, 3, 0]
    projections(X[:, selected], y, names,
                ['Petal length (cm)', 'Petal width (cm)', 'Sepal length (cm)'],
                'part2_features.png', 'Part 2: Iris in original feature coordinates')



if __name__ == '__main__':
    main()
