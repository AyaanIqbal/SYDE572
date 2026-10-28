"""Reproduce Assignment 3, Part 4. Run from any working directory."""
import json

from part2 import ROOT, COLORS
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA


def main():
    for folder in ['plots', 'results']:
        (ROOT / folder).mkdir(exist_ok=True)
    X = load_iris().data
    # Match Part 3: center original measurements without standardizing.
    pca = PCA(n_components=4, svd_solver='full')
    scores = pca.fit_transform(X)
    ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratio)
    np.savetxt(
        ROOT / 'results/explained_variance.csv',
        np.column_stack([np.arange(1, 5), pca.explained_variance_, ratio, cumulative]),
        delimiter=',', header='PC,eigenvalue,variance_ratio,cumulative_ratio', comments='')

    centered = X - pca.mean_
    reconstructed = scores[:, :2] @ pca.components_[:2] + pca.mean_
    total_ss = float(np.sum(centered**2))
    residual_ss = float(np.sum((X - reconstructed)**2))
    # Explained variance = 1 - residual energy / total centered energy.
    retained_from_error = 1 - residual_ss / total_ss
    np.testing.assert_allclose(retained_from_error, cumulative[1])
    np.testing.assert_allclose(total_ss, (len(X) - 1) * pca.explained_variance_.sum())

    summary = {
        'explained_variance_ratio': ratio.tolist(),
        'cumulative_variance_ratio': cumulative.tolist(),
        'two_pc_retained_fraction': float(cumulative[1]),
        'pcs_for_50_percent': int(np.searchsorted(cumulative, .5) + 1),
        'centered_frobenius_squared_cm2': total_ss,
        'two_pc_residual_squared_cm2': residual_ss,
        'two_pc_rmse_per_feature_cm': float(np.sqrt(residual_ss / X.size)),
    }
    (ROOT / 'results/part4_compression.json').write_text(
        json.dumps(summary, indent=2) + '\n')

    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    ax.bar(range(1, 5), ratio * 100, color=COLORS[0], label='Individual variance')
    ax.plot(range(1, 5), cumulative * 100, 'o-', color=COLORS[1],
            label='Cumulative variance')
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, label='50% target')
    ax.set(xticks=range(1, 5), xlabel='Principal component',
           ylabel='Explained variance (%)', ylim=(0, 106),
           title='Part 4: Variance retained by PCA')
    ax.legend()
    fig.savefig(ROOT / 'plots/part4_variance.png', dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
