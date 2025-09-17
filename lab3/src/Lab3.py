"""
ESE 388 - Fall 2025
Lab 3: Data Exploration & Clustering
Faculty: Vibha Mane
Student: Muhammad Sharjeel & Naafiul Hossain
"""

#******************************************************************************
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import warnings
warnings.filterwarnings("ignore")

#******************************************************************************
# Load Datasets
# Replace with correct file paths for Raisin.csv and DeepSpace.csv
raisin = pd.read_csv("Raisin.csv")
deep_space = pd.read_csv("DeepSpace.csv")

print("Raisin Dataset:")
print(raisin.head(), "\n")
print("Deep Space Dataset:")
print(deep_space.head(), "\n")

#******************************************************************************
# 1. Data Visualization: Raisin Dataset
print("### Raisin Data Exploration ###")

# Pairplot (scatter plots with hue = Class)
sns.pairplot(data=raisin, hue="Class", palette="husl")
plt.suptitle("Raisin Dataset - Pairplot by Class", y=1.02, fontsize=16)
plt.show()

# Histograms
raisin.hist(bins=20, figsize=(12, 8), edgecolor="black")
plt.suptitle("Raisin Dataset - Histograms", fontsize=16)
plt.show()

# Raisin dataset info
print("Class counts:\n", raisin["Class"].value_counts())
print("Columns:\n", raisin.columns)

#******************************************************************************
# 2. Data Visualization: Deep Space Dataset
print("### Deep Space Data Exploration ###")

# Pairplot (no class labels available)
sns.pairplot(data=deep_space, palette="coolwarm")
plt.suptitle("Deep Space Dataset - Pairplot (Unlabeled)", y=1.02, fontsize=16)
plt.show()

# Histograms
deep_space.hist(bins=20, figsize=(12, 8), edgecolor="black")
plt.suptitle("Deep Space Dataset - Histograms", fontsize=16)
plt.show()

#******************************************************************************
# 3. K-means Clustering on Raisin Dataset
print("### K-Means Clustering on Raisin Dataset ###")

# Separate features and target
raisinX = raisin.drop(columns=["Class"])
raisinY = raisin["Class"]

# Try different cluster counts
for k in [2, 3, 4, 5]:
    print(f"\nRunning K-Means on Raisin with k = {k}")
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(raisinX)
    raisin["Cluster"] = km.labels_

    dbi = davies_bouldin_score(raisinX, km.labels_)
    print(f"Davies-Bouldin Index for k = {k}: {dbi:.4f}")

    sns.pairplot(data=raisin, hue="Cluster", palette="coolwarm")
    plt.suptitle(f"Raisin - Pairplot with k = {k}", y=1.02, fontsize=16)
    plt.show()

#******************************************************************************
# 4. K-means Clustering on Deep Space Dataset
print("### K-Means Clustering on Deep Space Dataset ###")

deep_spaceX = deep_space.copy()

for k in [2, 3, 4, 5]:
    print(f"\nRunning K-Means on Deep Space with k = {k}")
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(deep_spaceX)
    deep_space["Cluster"] = km.labels_

    dbi = davies_bouldin_score(deep_spaceX, km.labels_)
    print(f"Davies-Bouldin Index for k = {k}: {dbi:.4f}")

    sns.pairplot(data=deep_space, hue="Cluster", palette="coolwarm")
    plt.suptitle(f"Deep Space - Pairplot with k = {k}", y=1.02, fontsize=16)
    plt.show()
