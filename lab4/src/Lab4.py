#Naafiul Hossain
#LAB4 DSB Clusters


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    confusion_matrix
)
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------
# Load data
# ---------------------------------------------
raisin = pd.read_csv("Raisin_Dataset.csv")
deep_space = pd.read_csv("DeepSpaceData.csv")

print("Raisin columns:", list(raisin.columns))
print("Deep Space columns:", list(deep_space.columns), "\n")

# ---------------------------------------------
# Helpers
# ---------------------------------------------
def standardize(df):
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)
    return X, scaler

def k_distance_eps_candidates(X, k=4):
    """
    Heuristic eps candidates from the distribution of the k-NN distances.
    Returns a sorted list of percentiles as candidate eps values.
    """
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    # distance to the kth neighbor for each point
    kth = np.sort(dists[:, -1])
    # pick eps around the upper percentiles (elbow-ish region)
    percentiles = [70, 75, 80, 85, 90, 92, 95]
    candidates = [np.percentile(kth, p) for p in percentiles]
    return sorted(set(np.round(candidates, 3))), kth

def dbscan_eval(X, labels):
    """
    Compute metrics on NON-NOISE points only.
    Returns dict with metrics and mask.
    """
    mask = labels != -1
    n_total = len(labels)
    n_core = mask.sum()
    n_clusters = len(set(labels[mask]))  # excludes noise

    out = {
        "n_total": n_total,
        "n_clustered": int(n_core),
        "coverage": n_core / n_total,
        "n_clusters": n_clusters,
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan
    }
    if n_clusters >= 2 and n_core >= 2:
        Xc = X[mask]
        yc = labels[mask]
        out["silhouette"] = float(silhouette_score(Xc, yc, metric="euclidean"))
        out["davies_bouldin"] = float(davies_bouldin_score(Xc, yc))
        out["calinski_harabasz"] = float(calinski_harabasz_score(Xc, yc))
    return out, mask

def best_dbscan(X, min_samples_grid=(3,4,5,6,8,10), k_for_eps=4):
    eps_list, kth = k_distance_eps_candidates(X, k=k_for_eps)
    print(f"Candidate eps (k={k_for_eps}): {eps_list}")

    results = []
    best = None

    for ms in min_samples_grid:
        for eps in eps_list:
            model = DBSCAN(eps=eps, min_samples=ms)
            labels = model.fit_predict(X)
            ev, _ = dbscan_eval(X, labels)
            ev.update({"eps": float(eps), "min_samples": int(ms)})
            results.append(ev)

    # Choose by highest silhouette; tie-break by more coverage, then more clusters
    dfres = pd.DataFrame(results)
    dfres["silhouette_rank"] = dfres["silhouette"].rank(method="min", ascending=False)
    dfres["coverage_rank"]   = dfres["coverage"].rank(method="min", ascending=False)
    dfres["clusters_rank"]   = dfres["n_clusters"].rank(method="min", ascending=False)
    dfres["score"] = (1000 - dfres["silhouette_rank"])*1e6 + (1000 - dfres["coverage_rank"])*1e3 + (1000 - dfres["clusters_rank"])

    choice = dfres.sort_values(["silhouette_rank","coverage_rank","clusters_rank"]).iloc[0]
    best_params = {"eps": float(choice["eps"]), "min_samples": int(choice["min_samples"])}

    # Fit final model
    final = DBSCAN(**best_params).fit(X)
    labels = final.labels_
    metrics, mask = dbscan_eval(X, labels)
    metrics.update(best_params)

    return final, labels, metrics, dfres.sort_values("silhouette_rank")

def pca_scatter(X, labels, title):
    p = PCA(n_components=2, random_state=42)
    X2 = p.fit_transform(X)
    dfplot = pd.DataFrame(X2, columns=["PC1","PC2"])
    dfplot["Cluster"] = labels

    plt.figure(figsize=(7,5))
    # noise as -1 (gray); clusters colored
    palette = sns.color_palette(n_colors=len(set(labels)) + 1)
    sns.scatterplot(
        data=dfplot, x="PC1", y="PC2",
        hue="Cluster", style="Cluster",
        palette="tab10", alpha=0.8
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()

def match_clusters_to_classes(true_labels, cluster_labels):
    """
    Map cluster IDs to class IDs (ignore noise) using Hungarian algorithm
    to maximize accuracy on clustered points.
    Returns mapping dict, accuracy on clustered points, and coverage.
    """
    mask = cluster_labels != -1
    y = true_labels[mask]
    c = cluster_labels[mask]
    if len(np.unique(c)) == 0:
        return {}, 0.0, 0.0

    classes = np.unique(y)
    clusters = np.unique(c)

    # Build confusion
    cm = confusion_matrix(y, c, labels=classes)
    # cost matrix for Hungarian (maximize correct => minimize negative)
    cost = cm.max() - cm
    r, col = linear_sum_assignment(cost)

    mapping = {clusters[j]: classes[i] for i, j in zip(r, col)}  # cluster -> class

    # Compute accuracy on clustered points
    y_pred = np.array([mapping.get(lbl, None) for lbl in c])
    acc = (y_pred == y).mean()

    coverage = mask.mean()
    return mapping, float(acc), float(coverage)

# ======================================================================
# 1) Raisin – DBSCAN
# ======================================================================
print("\n=== DBSCAN on Raisin (labeled) ===")
raisinX = raisin.drop(columns=["Class"])
raisinY = raisin["Class"].values
le = LabelEncoder()
y_true = le.fit_transform(raisinY)

Xr, _ = standardize(raisinX)

model_r, labels_r, metrics_r, grid_r = best_dbscan(Xr, min_samples_grid=(3,4,5,6,8,10), k_for_eps=4)
print("Chosen params (Raisin):", {k:metrics_r[k] for k in ("eps","min_samples")})
print("Metrics (non-noise only):", {k:round(metrics_r[k],4) for k in ("coverage","n_clusters","silhouette","davies_bouldin","calinski_harabasz")})

# Align clusters to classes
mapping_r, acc_r, cov_r = match_clusters_to_classes(y_true, labels_r)
print(f"Raisin: accuracy on clustered points = {acc_r:.4f}, coverage = {cov_r:.4f}")
if mapping_r:
    inv_classes = {i:cls for i, cls in enumerate(le.classes_)}
    pretty_map = {int(k): inv_classes[int(v)] for k, v in mapping_r.items()}
    print("Cluster→Class mapping:", pretty_map)

# Plot PCA
pca_scatter(Xr, labels_r, f"Raisin – DBSCAN (eps={metrics_r['eps']}, min_samples={metrics_r['min_samples']})")

# Optional: show top grid results
print("\nTop DBSCAN settings for Raisin by silhouette (showing first 8):")
print(grid_r.head(8)[["eps","min_samples","n_clusters","coverage","silhouette"]])

# ======================================================================
# 2) Deep Space – DBSCAN
# ======================================================================
print("\n=== DBSCAN on Deep Space (unlabeled) ===")
Xd, _ = standardize(deep_space)

model_d, labels_d, metrics_d, grid_d = best_dbscan(Xd, min_samples_grid=(3,4,5,6,8,10), k_for_eps=4)
print("Chosen params (Deep Space):", {k:metrics_d[k] for k in ("eps","min_samples")})
print("Metrics (non-noise only):", {k:round(metrics_d[k],4) for k in ("coverage","n_clusters","silhouette","davies_bouldin","calinski_harabasz")})

# PCA plot
pca_scatter(Xd, labels_d, f"Deep Space – DBSCAN (eps={metrics_d['eps']}, min_samples={metrics_d['min_samples']})")

# Optional: show a quick summary table of cluster sizes (including noise)
def cluster_summary(labels):
    vals, counts = np.unique(labels, return_counts=True)
    return pd.DataFrame({"label": vals, "count": counts}).sort_values("label").reset_index(drop=True)

print("\nCluster size summary – Raisin:")
print(cluster_summary(labels_r))
print("\nCluster size summary – Deep Space:")
print(cluster_summary(labels_d))
