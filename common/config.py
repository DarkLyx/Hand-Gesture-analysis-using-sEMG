import os

# --- CHOIX DE L'APPROCHE ---
MODE = "DL"  # "ML" ou "DL"

# --- CHEMINS ---
ROOT_DIR = 'data'             
CSV_DIR = 'csv_data'          
TENSOR_DIR = 'tensors_data'  
RESULTS_DIR = 'results'      

# Nom du fichier CSV principal
CSV_FILE = os.path.join(CSV_DIR, 'features_ml.csv')

# --- PARAMÈTRES DU SIGNAL ---
FS = 1000             # Fréquence d'échantillonnage (1000 Hz)
WINDOW_SIZE = 200     # On regarde 200ms de signal à la fois
STEP_SIZE = 50        # On avance de 50ms (recouvrement)
SAMPLES_TO_TRIM = 150 # On jette les 150 premiers/derniers points (transitions floues)
IGNORED_CLASSES = [0,7] # On ne veut pas classer le repos (0)

# --- FILTRES ---
LOW_CUT = 20.0        # Coupe les mouvements du corps (<20Hz)
HIGH_CUT = 450.0      # Coupe le bruit thermique (>450Hz)
NOTCH_FREQ = 50.0     # Coupe le bruit du réseau électrique (50Hz)
# --- SPLIT DATASET ---
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

# --- PARAMÈTRES CNN ---
BATCH_SIZE = 64
EPOCHS = 40