# Import Libraries
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Load Model, Scaler, and Label Encoder
model = joblib.load("models/galaxy_model.pkl")
scaler = joblib.load("models/scaler.pkl")
le = joblib.load("models/label_encoder.pkl")

# Load Dataset
df = pd.read_csv("data/galaxy_data.csv", comment='#', low_memory=False)
df.columns = df.columns.str.strip() 

features = [
    'u','g','r','i','z',
    'modelFlux_u','modelFlux_g','modelFlux_r','modelFlux_i','modelFlux_z',
    'petroFlux_u','petroFlux_g','petroFlux_r','petroFlux_i','petroFlux_z',
    'petroR50_u','petroR50_g','petroR50_r','petroR50_i','petroR50_z',
    'psfMag_u','psfMag_g','psfMag_r','psfMag_i','psfMag_z',
    'expAB_u','expAB_g','expAB_r','expAB_i','expAB_z'
]

X = df[features]
y = le.transform(df['subclass'])

# Feature Importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': features,
    'importance': importances
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title("Feature Importance - Random Forest")
plt.show()

print("Top 10 important features:")
print(feature_importance_df.head(10))

# PCA for Visualization
scaler_X = scaler.transform(X)  
pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaler_X)

plt.figure(figsize=(8,6))
for subclass in le.classes_:
    idx = df['subclass'] == subclass
    plt.scatter(X_pca[idx,0], X_pca[idx,1], label=subclass, alpha=0.5)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA of Galaxy Features")
plt.legend()
plt.show()

# Correlation with Target
numeric_features = features
corr = df[numeric_features].corrwith(pd.Series(y))
corr_df = pd.DataFrame({'feature': numeric_features, 'correlation_with_target': corr})
corr_df = corr_df.sort_values(by='correlation_with_target', key=abs, ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(x='correlation_with_target', y='feature', data=corr_df)
plt.title("Feature Correlation with Target (Absolute Value)")
plt.show()
