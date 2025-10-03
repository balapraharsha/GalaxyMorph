# 🌌 GalaxyMorph

AI-powered Python/Flask application for automated classification of galaxy morphology from photometric data.<br>
Classifies galaxies into **STARBURST**,**STARFORMING**, and other subclasses using a Random Forest Classifier.

----

## 🗂️ Project Structure
```bash

GalaxyMorph/
├── Python Executable Files/      # Main project folder
│   ├── app/                      # Web deployment component
│   │   ├── app.py                # Flask application
│   │   └── templates/            # HTML templates
│   │       └── index.html
│   ├── data/                     # Dataset folder
│   │   └── galaxy_data.csv       # SDSS photometric features
│   ├── models/                   # Saved model artifacts
│   │   ├── galaxy_model.pkl      # Trained Random Forest model
│   │   ├── label_encoder.pkl     # Label encoder
│   │   └── scaler.pkl            # Scaler for feature normalization
│   ├── notebooks/                # Jupyter notebooks for EDA & analysis
│   │   ├── data_exploration.py
│   │   ├── feature_analysis.py
│   │   └── model_training.py
│   └── results/                  # Results & visualizations
│       ├── data_exploration/
│       │   ├── flux_feature_distributions.png
│       │   ├── galaxy_morphology_distribution.png
│       │   └── ...
│       ├── model_training/
│       │   ├── classification_report.txt
│       │   ├── confusion_matrix.png
│       │   └── feature_importance.png
│       └── data_summary/
│           └── feature_min_max.csv
├── Project Demonstration.mp4     # Video demo
├── Project Documentation.pdf     # Project report
├── README.md                     # Project overview & instructions
└── .gitattributes                # Git attributes file



```

## ✨ Features
- Automated galaxy morphology classification from SDSS photometric features
- Confidence scores & probability distribution
- Interactive Flask web app for real-time predictions
- EDA, feature importance visualization, and model evaluation outputs  

## 🛠️ Tech Stack
- Python 3.11
- scikit-learn (Random Forest)
- Flask (Web Framework)
- Pandas, Numpy, Matplotlib, Seaborn

## 🚀 How to Run

**Clone the repository**
```bash
git clone https://github.com/balapraharsha/GalaxyMorph.git
cd GalaxyMorph/Python\ Executable\ Files
```
---

**Create virtual environment & activate**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```
---

**Install dependencies**
```bash
pip install -r requirements.txt
```
---

**Run the Flask app**
```bash
python app/app.py
```
---

**Open in browser**
- As given local host


----

## 🖥️ Functionalities
- Upload galaxy features manually or from CSV
- Predict galaxy subclass (STARBURST, STARFORMING, etc.)
- View predicted class & confidence score
- Visualize feature distributions and model outputs

---

## 📊 Model Details
- Algorithm: Random Forest Classifier
- Input: Photometric features (u, g, r, i, z, fluxes, radii, PSF mags, etc.)
- Output: Galaxy subclass
- Evaluation:
    - Accuracy: 88%
    - Precision / Recall / F1-score: STARBURST 0.75 / 0.71 / 0.75, STARFORMING 0.92 / 0.93 / 0.92
- Model artifacts: scaler, label encoder, trained RF model

---
## 🗃️ Dataset Overview
- Source: SDSS photometric survey
- Shape: (100000, 43) features including magnitudes, fluxes, and radii
- Classes: STARBURST, STARFORMING, AGN, etc.

---

## 📈 Insights Summary
- Random Forest performs well for photometric feature-based galaxy classification
- Feature importance helps understand which bands and fluxes drive predictions
- Web app allows easy real-time galaxy morphology prediction

---

## 🌟 Highlights
- Full-stack AI project: ML + Web App
- Real-time predictions with confidence
- EDA & visualization for insights
- Easy-to-use interface for researchers & students

---

## 👨‍💻 Developed By

**Bala Praharsha .M**  
📧 [balapraharsha.m@gmail.com]  
🔗 [LinkedIn](https://linkedin.com/in/mannepalli-bala-praharsha) | [GitHub](https://github.com/balapraharsha)  

---

## 💖 Show Some Love
Enjoying this project? Give it a **star** ⭐ on GitHub!  
Contributions, suggestions, and forks are always welcome.

---
