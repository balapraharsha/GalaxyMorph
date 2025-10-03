import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create results folder
os.makedirs("results/data_exploration", exist_ok=True)

# Load dataset
df = pd.read_csv("data/galaxy_data.csv", comment='#', low_memory=False)
df.columns = df.columns.str.strip() 

print("Dataset shape:", df.shape)
print(df.head())

# Basic Info
print(df.info())
print(df.describe())

# Handle missing/invalid values
df.replace(-9999, np.nan, inplace=True)
print("Missing values per column:\n", df.isnull().sum())

# Target Distribution
print("Target (subclass) distribution:\n", df['subclass'].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x='subclass', data=df, order=df['subclass'].value_counts().index, palette='pastel')
plt.title("Galaxy Morphology Distribution")
plt.savefig("results/data_exploration/galaxy_morphology_distribution.png")
plt.show()

# Feature Distribution (photometric magnitudes)
photometric_features = ['u','g','r','i','z']

df[photometric_features].hist(bins=30, figsize=(12,6), color='skyblue', edgecolor='black')
plt.suptitle("Photometric Magnitude Distributions")
plt.savefig("results/data_exploration/photometric_magnitude_distributions.png")
plt.show()

# Flux Feature Distribution
flux_features = [
    'modelFlux_u','modelFlux_g','modelFlux_r','modelFlux_i','modelFlux_z',
    'petroFlux_u','petroFlux_g','petroFlux_r','petroFlux_i','petroFlux_z'
]

df[flux_features].hist(bins=30, figsize=(14,6), color='lightgreen', edgecolor='black')
plt.suptitle("Flux Feature Distributions")
plt.savefig("results/data_exploration/flux_feature_distributions.png")
plt.show()

# Scatter plots: Radii vs r magnitude
radii_features = ['petroR50_u','petroR50_g','petroR50_r','petroR50_i','petroR50_z']

plt.figure(figsize=(12,6))
for i, col in enumerate(radii_features):
    plt.subplot(2,3,i+1)
    plt.scatter(df[col], df['r'], alpha=0.5, s=10, color='orange')
    plt.xlabel(col)
    plt.ylabel('r magnitude')
plt.tight_layout()
plt.suptitle("Radii vs r Magnitude", y=1.02)
plt.savefig("results/data_exploration/radii_vs_r_magnitude.png")
plt.show()

# Correlation heatmap
numeric_features = photometric_features + flux_features + radii_features
corr = df[numeric_features].corr()

plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title("Feature Correlation Heatmap")
plt.savefig("results/data_exploration/feature_correlation_heatmap.png")
plt.show()

# Pairplot (sample of 500 rows)
sample_df = df.sample(n=500, random_state=42)
pairplot_fig = sns.pairplot(sample_df[photometric_features + ['subclass']], hue='subclass', corner=True, palette='bright')
pairplot_fig.fig.suptitle("Pairplot of Photometric Features", y=1.02)
pairplot_fig.savefig("results/data_exploration/pairplot_photometric_features.png")
plt.show()
