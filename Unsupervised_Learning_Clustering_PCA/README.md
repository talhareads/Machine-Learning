# 🔮 Unsupervised Learning — Clustering & Dimensionality Reduction

A hands-on exploration of unsupervised machine learning techniques: **K-Means clustering**, **DBSCAN (density-based clustering)**, and **PCA (dimensionality reduction)**, using synthetic datasets to clearly visualize how each algorithm behaves.

## 📌 Project Overview

Unlike supervised learning, unsupervised learning finds structure in data **without labeled outcomes**. This project walks through three core unsupervised techniques, each demonstrated on a dataset specifically chosen to highlight its strengths (and, in the case of K-Means, its limitations).

**Workflow:**
1. **K-Means Clustering** on globular (blob-shaped) data — with the Elbow Method to find the optimal number of clusters
2. **DBSCAN vs. K-Means** comparison on non-globular (moon-shaped) data — showing why density-based clustering handles complex shapes better
3. **PCA (Principal Component Analysis)** — reducing 5-dimensional data down to 2 dimensions for visualization

## 📊 Datasets

All datasets are **synthetically generated** using Scikit-learn, so no external files are needed:

| Section | Generator | Description |
|---|---|---|
| K-Means | `make_blobs(n_samples=500, centers=3, cluster_std=0.60)` | 3 well-separated circular clusters |
| DBSCAN vs K-Means | `make_moons(n_samples=500, noise=0.05)` | 2 interleaving crescent (moon) shapes |
| PCA | `make_blobs(n_samples=500, n_features=5, centers=3, cluster_std=0.6)` | 3 clusters in 5-dimensional space |

All features were standardized using `StandardScaler` before clustering/PCA.

## 🤖 Techniques & Results

### 1️⃣ K-Means Clustering
- Used the **Elbow Method** (testing `k=1` to `10`) to determine the optimal number of clusters.
- Inertia dropped sharply from **1000.0 (k=1)** to **11.57 (k=3)**, then flattened out — confirming **k=3** as the optimal cluster count, matching the true number of blob centers.
- Final model: `KMeans(n_clusters=3)`, visualized with a scatter plot colored by cluster label.

### 2️⃣ DBSCAN vs. K-Means (Density-Based Clustering)
- On the **moon-shaped dataset**, K-Means (`n_clusters=2`) was applied first — but since K-Means assumes spherical clusters, it struggles to correctly separate the crescent shapes.
- **DBSCAN** (`eps=0.2`, `min_samples=5`) was applied next, correctly identifying the two crescent-shaped clusters based on point density rather than distance to a centroid.
- This comparison visually demonstrates why **density-based clustering outperforms centroid-based clustering on non-convex shapes**.

### 3️⃣ PCA (Dimensionality Reduction)
- Generated a 5-feature dataset with 3 underlying clusters — impossible to visualize directly.
- Applied **PCA** to reduce the 5 dimensions down to **2 principal components (PC1, PC2)**.
- Plotted the reduced data in 2D, colored by the true cluster label, showing that PCA preserved enough variance to keep the 3 clusters clearly separable even after dimensionality reduction.

## 🛠️ Tech Stack

- **Python**
- **Pandas** — data handling
- **Matplotlib / Seaborn** — cluster & PCA visualizations
- **Scikit-learn** — `make_blobs`, `make_moons`, `KMeans`, `DBSCAN`, `PCA`, `StandardScaler`

## 📁 Project Structure

```
unsupervised-learning-clustering-pca/
│
├── Unsupervised_Learning_Clustering_PCA.ipynb   # K-Means, DBSCAN & PCA implementation
└── README.md
```

## 🚀 Future Improvements

- Apply these techniques to a real-world dataset (e.g. customer segmentation) instead of synthetic data
- Evaluate clustering quality quantitatively using Silhouette Score or Adjusted Rand Index (vs. true labels)
- Tune DBSCAN's `eps` and `min_samples` systematically instead of manually
- Explore additional dimensionality reduction techniques (t-SNE, UMAP) and compare against PCA

## 👤 Author

**Talha Rashid**

*With help and guidance from **Zahid Hashmi***

---
⭐ If you found this project useful, consider giving it a star!
