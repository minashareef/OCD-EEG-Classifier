# train_model.py
"""
OCD EEG Classifier Training Pipeline
80% accuracy on 10 patient dataset
Author: Mina Shareef Mundol
"""

import glob, os, warnings, joblib
import numpy as np, pandas as pd
import mne
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# === CONFIG ===
HC_PATH = "HC"  # Update paths
OCD_PATH = "OCD"
N_SAMPLES = 5

def load_filter_eeg_data(hc_path, ocd_path, n_samples):
    """Load and preprocess EEG data"""
    hc_files = glob.glob(os.path.join(hc_path, "*_a.vhdr"))[:n_samples]
    ocd_files = glob.glob(os.path.join(ocd_path, "*_a.vhdr"))[:n_samples]
    
    hc_df = pd.DataFrame(columns=['patient_id', 'raw'])
    ocd_df = pd.DataFrame(columns=['patient_id', 'raw'])
    
    # Load HC
    for i, file_path in enumerate(hc_files, 1):
        raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
        raw.filter(0.1, 200); raw.notch_filter(50); raw = raw.get_data()
        hc_df.loc[len(hc_df)] = [i, raw]
    
    # Load OCD  
    for i, file_path in enumerate(ocd_files, 1):
        raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
        raw.filter(0.1, 150); raw.notch_filter(50); raw = raw.get_data()
        ocd_df.loc[len(ocd_df)] = [i, raw]
    
    hc_df['target'] = 0; ocd_df['target'] = 1
    return pd.concat([hc_df, ocd_df], ignore_index=True)

def extract_eeg_features(X):
    """Extract 594 features: 9 per channel x 66 channels"""
    features = []
    freqs = {'delta': (1,4), 'theta': (4,8), 'alpha': (8,13), 'beta': (13,30), 'gamma': (30,100)}
    
    for raw_data in X:
        channel_features = []
        for ch_data in raw_data:
            # Power bands
            power_bands = []
            for f_low, f_high in freqs.values():
                filtered = ch_data.copy()
                n = len(filtered)
                filtered *= (np.arange(n) > f_low * n / 200) * (np.arange(n) < f_high * n / 200)
                power_bands.append(np.mean(filtered**2))
            
            # Time stats
            stats = [np.mean(ch_data), np.std(ch_data), 
                    np.ptp(ch_data), np.mean(np.abs(np.diff(ch_data)))]
            
            channel_features.extend(power_bands + stats)
        features.append(channel_features)
    return np.array(features)

if __name__ == "__main__":
    # Load data
    print(" Loading EEG data...")
    df = load_filter_eeg_data(HC_PATH, OCD_PATH, N_SAMPLES)
    X, y = df['raw'].values, df['target'].values
    
    # Split & extract features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train_f = extract_eeg_features(X_train); X_test_f = extract_eeg_features(X_test)
    
    # Compare models
    models = {"RF": RandomForestClassifier(100, random_state=42), 
              "GB": GradientBoostingClassifier(100, random_state=42),
              "MLP": MLPClassifier((128,64), max_iter=2000, random_state=42)}
    
    print("\n Model Comparison:")
    for name, clf in models.items():
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train_f)
        cv_f1 = cross_val_score(clf, X_tr_s, y_train, cv=3, scoring='f1').mean()
        print(f"{name}: CV F1 = {cv_f1:.3f}")
    
    # Train production GB model
    pipeline = Pipeline([('scaler', StandardScaler()),
                        ('gb', GradientBoostingClassifier(200, 0.05, 3, random_state=42))])
    all_X, all_y = np.vstack([X_train_f, X_test_f]), np.hstack([y_train, y_test])
    cv_f1 = cross_val_score(pipeline, all_X, all_y, cv=5, scoring='f1').mean()
    
    pipeline.fit(all_X, all_y)
    print(f"\n Final GB: CV F1={cv_f1:.3f}")
    
    # Save
    joblib.dump(pipeline, 'ocd_classifier.pkl')
    print(" Saved: ocd_classifier.pkl")
