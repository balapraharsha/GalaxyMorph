import os
import pandas as pd
df = pd.read_csv("data/galaxy_data.csv", comment='#', low_memory=False)
df.columns = df.columns.str.strip() 
# Define the features you care about
features = [
    'u','g','r','i','z',
    'modelFlux_u','modelFlux_g','modelFlux_r','modelFlux_i','modelFlux_z',
    'petroFlux_u','petroFlux_g','petroFlux_r','petroFlux_i','petroFlux_z',
    'petroR50_u','petroR50_g','petroR50_r','petroR50_i','petroR50_z',
    'psfMag_u','psfMag_g','psfMag_r','psfMag_i','psfMag_z',
    'expAB_u','expAB_g','expAB_r','expAB_i','expAB_z'
]

# Keep only the columns that exist in df
valid_features = [col for col in features if col in df.columns]

print("✅ Valid features found:", valid_features)
print("⚠️ Missing features:", set(features) - set(valid_features))

# Compute min and max
min_max_df = df[valid_features].agg(['min', 'max']).T
min_max_df.columns = ['Min Value', 'Max Value']

# Save results
os.makedirs("results/data_summary", exist_ok=True)
min_max_df.to_csv("results/data_summary/feature_min_max.csv", index=True)

print("📂 Saved min/max values to results/data_summary/feature_min_max.csv")
print(min_max_df.head())
