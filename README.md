# OCD-EEG-Classifier
80% accurate ML model to detect OCD from EEG
#  OCD EEG Classifier (80% Accuracy!)

Detects OCD vs Healthy Control from EEG data using Gradient Boosting.

##  Results
- **CV Accuracy:** 80.0%
- **CV F1 Score:** 73.3% 
- **Key Biomarkers:** Gamma power in Channels 22, 51, 55

##  Quick Start
```bash
pip install -r requirements.txt
python predict_new_patient.py your_eeg.vhdr
