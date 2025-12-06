from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def train_ml_model(X_train, y_train):
    """Définit et entraîne le pipeline SVM via GridSearch."""
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),       # Normalisation
        ('lda', LDA()),                     # Réduction de dimension supervisée
        ('svm', SVC(kernel='rbf', class_weight='balanced')) # Classification
    ])
    
    param_grid = {
        'svm__C': [1, 10, 100],
        'svm__gamma': ['scale', 0.01, 0.001]
    }
    
    print("   [ML] Optimisation GridSearch en cours...")
    grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    
    print(f"   [ML] Meilleurs paramètres : {grid.best_params_}")
    return grid.best_estimator_