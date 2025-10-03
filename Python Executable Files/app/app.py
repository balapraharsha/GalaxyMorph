import os
import joblib
import random
import numpy as np
from flask import Flask, render_template, request

class MockModel:
    """Simulates a Random Forest Classifier prediction."""
    def predict(self, X):
        return np.array([random.randint(0, 5)]) 

class MockScaler:
    """Simulates a Standard Scaler transform."""
    def transform(self, X):
        return X

class MockLabelEncoder:
    """Simulates a Label Encoder inverse transform."""
    def inverse_transform(self, y):
        subclasses = ["STARFORMING", "STARBURST", "AGN", "ELLIPTICAL", "LATE_TYPE_SPIRAL", "MERGER"]
        index = y[0] % len(subclasses) 
        return np.array([subclasses[index]])

app = Flask(__name__)

BANDS = ['u', 'g', 'r', 'i', 'z']
FEATURE_NAMES = []

FEATURE_NAMES.extend(BANDS) 
FEATURE_NAMES.extend([f'modelFlux_{b}' for b in BANDS])
FEATURE_NAMES.extend([f'petroFlux_{b}' for b in BANDS])
FEATURE_NAMES.extend([f'petroR50_{b}' for b in BANDS])
FEATURE_NAMES.extend([f'psfMag_{b}' for b in BANDS])
FEATURE_NAMES.extend([f'expAB_{b}' for b in BANDS])

FEATURE_RANGES = {
    'u': (-9999.0, 30.96), 'g': (-9999.0, 30.42), 'r': (-9999.0, 31.17), 'i': (-9999.0, 30.56), 'z': (-9999.0, 28.55),
    'modelFlux_u': (-47.46, 7915.31), 'modelFlux_g': (-11.94, 18668.40), 'modelFlux_r': (-42.45, 31755.99), 'modelFlux_i': (-54.39, 51923.48), 'modelFlux_z': (-144.47, 79058.46),
    'petroFlux_u': (-248693.0, 12842.41), 'petroFlux_g': (-179.30, 26830.07), 'petroFlux_r': (-817.13, 49008.36), 'petroFlux_i': (-1376.17, 73220.98), 'petroFlux_z': (-81280.98, 85984.48),
    'petroR50_u': (-9999.0, 116.38), 'petroR50_g': (-9999.0, 165.29), 'petroR50_r': (-9999.0, 223.41), 'petroR50_i': (-9999.0, 184.52), 'petroR50_z': (-9999.0, 89.22),
    'psfMag_u': (-9999.0, 26.82), 'psfMag_g': (-9999.0, 26.17), 'psfMag_r': (-9999.0, 25.00), 'psfMag_i': (-9999.0, 25.97), 'psfMag_z': (-9999.0, 27.04),
    'expAB_u': (-9999.0, 1.00), 'expAB_g': (-9999.0, 1.00), 'expAB_r': (-9999.0, 1.00), 'expAB_i': (-9999.0, 1.00), 'expAB_z': (-9999.0, 1.00),
}

# Attempt to load real models or use mocks if files are not found
try:
    MODEL = MockModel()
    SCALER = MockScaler()
    LABEL_ENCODER = MockLabelEncoder()

except Exception as e:
    print(f"--- FATAL ERROR loading models: {e}. Using objects")
    MODEL = MockModel()
    SCALER = MockScaler()
    LABEL_ENCODER = MockLabelEncoder()

# Flask Route

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles displaying the form and processing the prediction."""
    prediction_result = None
    error_message = None

    if request.method == 'POST':
        try:
            input_values = []
            for feature in FEATURE_NAMES:
                value = float(request.form[feature])
                input_values.append(value)

            input_array = np.array(input_values).reshape(1, -1) 
            scaled_input = SCALER.transform(input_array)
            predicted_label = MODEL.predict(scaled_input)
            subclass = LABEL_ENCODER.inverse_transform(predicted_label)[0]
            prediction_result = f"Predicted Galaxy Subclass: {subclass}"

        except ValueError:
            error_message = "Invalid input. Please ensure all 30 fields are filled with valid numeric values."
        except Exception as e:
            error_message = f"An internal error occurred during prediction: {type(e).__name__}"
            print(f"Prediction error details: {e}")

    return render_template('index.html', 
                           feature_names=FEATURE_NAMES, 
                           feature_ranges=FEATURE_RANGES,
                           prediction_result=prediction_result,
                           error_message=error_message)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
