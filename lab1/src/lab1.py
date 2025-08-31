# ESE 388 – Lab 1: Python Arrays & DataFrames
# Author: Naafiul Hossain
# Date: 8/27/25

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  # only to display/save the pairplot

# ---- Part 1: 1-column, 100-row NumPy array ----
np.random.seed(42)  # for reproducibility
arr_1col = np.random.randn(100, 1)  # shape: (100, 1)

# Convert to pandas Series
series_1col = pd.Series(arr_1col.ravel(), name="RandomValues")
print("Series (first 5 rows):")
print(series_1col.head(), "\n")

# ---- Part 2: 3-column, 100-row NumPy array ----
arr_3col = np.random.randn(100, 3)  # shape: (100, 3)

# Convert to pandas DataFrame with labels X1, X2, X3
df = pd.DataFrame(arr_3col, columns=["X1", "X2", "X3"])
print("DataFrame (first 5 rows):")
print(df.head(), "\n")

# ---- Part 3: Seaborn pairplot ----
sns.set(style="whitegrid")
g = sns.pairplot(df, diag_kind="hist")
plt.suptitle("Pairplot of X1, X2, X3 (n=100)", y=1.02)

# Show the plot (and optionally save it)
plt.show()
# g.savefig("pairplot.png", dpi=150)  # uncomment to save
