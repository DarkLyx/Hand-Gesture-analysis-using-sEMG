import os

# APPROACH CHOICE
MODE = "CNN"  # "ML", "CNN", "CNN-no-preprocessing", "CNN-VGG-transfert", "VIT" or "GNN"

#
USE_PREPROCESSING = (MODE != "CNN-no-preprocessing")

# 
ROOT_DIR = 'data'             
CSV_DIR = 'csv_data'          
TENSOR_DIR = 'tensors_data'  
RESULTS_DIR = 'results'      

# FEATURES FILE PATH (ML)
CSV_FILE = os.path.join(CSV_DIR, 'features_ml.csv')

# SIGNAL PARAMETERS
FS = 1000               # Sampling frequency (1000 Hz)
WINDOW_SIZE = 200       # Analyze 200 ms signal windows
STEP_SIZE = 50          # Sliding step of 50 ms (overlapping windows)
SAMPLES_TO_TRIM = 150   # Discard the first and last 150 samples (unstable transitions)
IGNORED_CLASSES = [0,7] # Exclude rest class (0) from classification

# FILTERS
LOW_CUT = 20.0          # Remove motion artifacts (< 20 Hz)
HIGH_CUT = 450.0        # Remove high-frequency noise (> 450 Hz)
NOTCH_FREQ = 50.0       # Suppress electrical grid noise (50 Hz)

# SPLIT DATASET
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

# CNN PARAMETERS
BATCH_SIZE = 64
EPOCHS = 40

# SPECTROGRAM
SPECTRO_NPERSEG = 32
SPECTRO_NOVERLAP = 16

# VIT PARAMETERS
PATCH_SIZE = 20         # Temporal patch size (ex: 20ms)
PROJECTION_DIM = 64     # Latent space dimension (Embedding)
NUM_HEADS = 4           # Number of attention head
TRANSFORMER_LAYERS = 4  # Network depth (Blocs number)
MLP_HEAD_UNITS = [2048, 1024] # Finales dense layers

# GNN PARAMETERS
GRAPH_CHANNELS = 8  # Number of nodes (sensors)
GNN_LAYERS = 3      # Depth