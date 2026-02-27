# predict_new_patient.py
"""
Predict OCD vs HC from new EEG file
Usage: python predict_new_patient.py path/to/eeg.vhdr
"""

import sys, joblib, mne, numpy as np
from train_model import extract_eeg_features  # Import feature extractor

# Load model
model = joblib.load('ocd_classifier.pkl')

def predict_single_eeg(eeg_file):
    """Full pipeline for single EEG file"""
    # Load & preprocess
    raw = mne.io.read_raw_brainvision(eeg_file, preload=True, verbose=False)
    raw.filter(0.1, 200); raw.notch_filter(50)
    eeg_data = raw.get_data()
    
    # Extract features & predict
    features = extract_eeg_features([eeg_data])
    pred = model.predict(features)[0]
    prob_ocd = model.predict_proba(features)[0, 1]
    
    label = " OCD DETECTED" if pred else " Healthy Control"
    return f"{label} (confidence: {prob_ocd:.1%})"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_new_patient.py your_eeg.vhdr")
        sys.exit(1)
    
    result = predict_single_eeg(sys.argv[1])
    print(result)
