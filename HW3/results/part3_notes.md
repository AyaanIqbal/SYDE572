PCA was applied to all four measurements after centering, without standardizing or whitening them. In the order sepal length, sepal width, petal length, and petal width, the mean is (5.843333, 3.057333, 3.758000, 1.199333) cm. Each flower’s score on axis j is calculated by projecting its centered measurements onto that axis: z_j = (x − μ) · w_j.

Each row of the table below gives a unit principal-component vector, and the four vectors are mutually orthogonal. An axis can point in either direction, so reversing its sign only reflects the plot without changing the information retained.

PC1 has its largest loading on petal length and clearly separates setosa from the other species. The first two PCs reveal the broad species groups, although versicolor and virginica still overlap. Adding PC3 preserves more variation without making those groups completely distinct, which is consistent with PCA’s aim of capturing variance rather than maximizing class separation.
