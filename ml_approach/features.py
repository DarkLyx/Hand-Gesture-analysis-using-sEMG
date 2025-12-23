import numpy as np
from scipy.signal import welch
import common.config as cfg

def calculate_features(segment):
    """Extraction Time Domain & Frequency Domain."""
    features = {}
    threshold = 1e-8 # Threshold for low/normalize signals
    
    for i in range(segment.shape[1]):
        sig = segment[:, i]
        ch = f'ch{i+1}'
        
        # Time Domain
        features[f'{ch}_MAV'] = np.mean(np.abs(sig))
        features[f'{ch}_RMS'] = np.sqrt(np.mean(sig**2))
        features[f'{ch}_WL'] = np.sum(np.abs(np.diff(sig)))
        features[f'{ch}_ZC'] = np.sum((sig[:-1] * sig[1:] < 0) & (np.abs(sig[:-1] - sig[1:]) > threshold))
        
        # Frequency Domain
        f, Pxx = welch(sig, cfg.FS, nperseg=len(sig))
        mnf = np.sum(f * Pxx) / np.sum(Pxx) if np.sum(Pxx) > 0 else 0
        features[f'{ch}_MNF'] = mnf
        
    return features