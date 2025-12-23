import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
import common.config as cfg

def notch_filter(data, fs=cfg.FS, freq=cfg.NOTCH_FREQ, Q=30.0):
    """Notch filter (50 Hz) to remove electrical grid noise."""
    nyq = 0.5 * fs
    freq_norm = freq / nyq
    b, a = iirnotch(freq_norm, Q)
    return filtfilt(b, a, data, axis=0)

def butter_bandpass_filter(data, order=4):
    """Band-pass filter between 20 and 450 Hz."""
    nyq = 0.5 * cfg.FS
    low = cfg.LOW_CUT / nyq
    high = cfg.HIGH_CUT / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

def clean_and_trim_data(df):
    """Remove ignored classes and trim transition samples."""
    df_clean = df[~df['class'].isin(cfg.IGNORED_CLASSES)].copy()
    if df_clean.empty: return None

    df_clean['group_id'] = (df_clean['class'] != df_clean['class'].shift(1)).cumsum()
    
    def trim_group(group):
        if len(group) <= 2 * cfg.SAMPLES_TO_TRIM: return None
        return group.iloc[cfg.SAMPLES_TO_TRIM : -cfg.SAMPLES_TO_TRIM]

    df_trimmed = df_clean.groupby('group_id').apply(trim_group).reset_index(drop=True)
    if 'group_id' in df_trimmed.columns: df_trimmed = df_trimmed.drop(columns=['group_id'])
    return df_trimmed

def get_windows(df):
    """Split the dataframe into sliding windows (generator)."""
    channels = [c for c in df.columns if 'channel' in c]
    for label, group in df.groupby('class'):
        data = group[channels].values
        for i in range(0, len(data) - cfg.WINDOW_SIZE + 1, cfg.STEP_SIZE):
            yield data[i : i + cfg.WINDOW_SIZE], label