# GalaxyMorph

GalaxyMorph is a machine learning-based Flask web application designed to classify galaxy images into subclasses such as STARBURST and STARFORMING using a trained Random Forest model. It combines data science with a simple web interface to provide fast and accurate galaxy morphology predictions.

## This project contains:

- **app/** – Flask application files  
  - `app.py` – Main Flask app  
  - `templates/` – HTML templates for user interface  

- **models/** – Trained models and encoders  
  - `galaxy_model.pkl` – Trained Random Forest model  
  - `scaler.pkl` – StandardScaler for feature scaling  
  - `label_encoder.pkl` – LabelEncoder for target classes  

- **data/** – Galaxy dataset  
  - `galaxy_data.csv` – Input dataset used for training  

- **notebooks/** – Jupyter notebooks for analysis and model training  
  - `data_exploration.py` – Exploratory Data Analysis  
  - `feature_analysis.py` – Feature importance and correlation  
  - `model_training.py` – Model training and evaluation  
  - `sample.py` – Sample scripts  

- **results/** – Output visualizations and reports  
  - `data_exploration/` – Plots for photometric features, fluxes, radii, etc.  
  - `data_summary/` – Feature min/max CSV files  
  - `model_training/` – Confusion matrix, classification report, feature importance  

- **requirements.txt** – Python dependencies  

---
   ```bash
   pip install -r requirements.txt
