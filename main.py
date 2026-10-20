import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Simulasi / Pembuatan Dummy Dataset (Merepresentasikan Ekstraksi Fitur Satelit Sentinel-1 & Sentinel-2)
np.random.seed(42)
n_samples = 1500

data = {
    'sentinel1_vh_mean': np.random.uniform(-25.0, -5.0, n_samples),
    'sentinel2_nir_mean': np.random.uniform(0.1, 0.9, n_samples),
    'sentinel2_swir_mean': np.random.uniform(0.05, 0.6, n_samples),
    'fire_hotspot_detected': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
    'patch_area_ha': np.exponential(scale=5.0, size=n_samples),
    'distance_to_road_km': np.exponential(scale=2.0, size=n_samples)
}

df = pd.DataFrame(data)

# Logika penentuan Target Driver tiruan berdasarkan karakteristik sensor
def assign_driver(row):
    if row['fire_hotspot_detected'] == 1 and row['sentinel2_swir_mean'] > 0.4:
        return 'Wildfire'
    elif row['patch_area_ha'] > 15.0:
        return 'Large-Scale Agriculture'
    elif row['distance_to_road_km'] < 1.0 and row['sentinel1_vh_mean'] > -12.0:
        return 'Selective Logging'
    elif row['sentinel1_vh_mean'] > -10.0 and row['patch_area_ha'] < 5.0:
        return 'Mining'
    else:
        return 'Small-Scale Agriculture'

df['target_driver'] = df.apply(assign_driver, axis=1)

# 2. Data Preparation & Train-Test Split
X = df.drop(columns=['target_driver'])
y = df['target_driver']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 3. Pemodelan (Modeling) menggunakan Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Prediksi & Evaluasi
y_pred = model.predict(X_test)
print("--- LAPORAN KLASIFIKASI MODEL ---")
print(classification_report(y_test, y_pred))

# Visualisasi Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_ if hasattr(model, 'classes_') else model.classes_)
plt.xlabel('Prediksi Model')
plt.ylabel('Kondisi Aktual (Ground Truth)')
plt.title('Confusion Matrix - Klasifikasi Penyebab Deforestasi')
plt.tight_layout()
plt.show()