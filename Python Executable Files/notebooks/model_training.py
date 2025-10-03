# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# Create results folder
os.makedirs("results/model_training", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load Dataset
df = pd.read_csv("data/galaxy_data.csv", comment='#', low_memory=False)
df.columns = df.columns.str.strip()
print("Dataset shape:", df.shape)
print(df.head())

# Select Features & Target
features = [
    'u','g','r','i','z',
    'modelFlux_u','modelFlux_g','modelFlux_r','modelFlux_i','modelFlux_z',
    'petroFlux_u','petroFlux_g','petroFlux_r','petroFlux_i','petroFlux_z',
    'petroR50_u','petroR50_g','petroR50_r','petroR50_i','petroR50_z',
    'psfMag_u','psfMag_g','psfMag_r','psfMag_i','psfMag_z',
    'expAB_u','expAB_g','expAB_r','expAB_i','expAB_z'
]

X = df[features]
y = df['subclass']

# Encode Target
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Classes:", le.classes_)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate Model
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")

# Save Accuracy to a text file
with open("results/model_training/test_accuracy.txt", "w") as f:
    f.write(f"Test Accuracy: {acc:.4f}\n")

# Classification Report
report = classification_report(y_test, y_pred, target_names=le.classes_)
print("\nClassification Report:")
print(report)

# Save Classification Report
with open("results/model_training/classification_report.txt", "w") as f:
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("results/model_training/confusion_matrix.png")
plt.show()

# Feature Importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': features,
    'importance': importances
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title("Feature Importance - Random Forest")
plt.savefig("results/model_training/feature_importance.png")
plt.show()

# Save Model, Scaler, and Label Encoder
joblib.dump(model, "models/galaxy_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(le, "models/label_encoder.pkl")

print("Model, scaler, and label encoder saved successfully!")
