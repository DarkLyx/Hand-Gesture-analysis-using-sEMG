import numpy as np
from sklearn.metrics import accuracy_score
import common.metrics as met

def optimize_and_apply_majority_voting(y_val_true, y_val_pred_raw, y_test_pred_raw):
    best_win, best_acc = 1, 0
    possible_windows = [1, 3, 5, 7, 9]
    
    for w in possible_windows:
        smooth_val = met.majority_voting(y_val_pred_raw, w)
        acc = accuracy_score(y_val_true, smooth_val)
        
        if acc > best_acc: 
            best_win, best_acc = w, acc
            
    print(f"   -> Best window found: {best_win} (Val Acc: {best_acc:.2%})")
    
    y_val_smooth = met.majority_voting(y_val_pred_raw, best_win)
    y_test_smooth = met.majority_voting(y_test_pred_raw, best_win)
    
    return y_val_smooth, y_test_smooth, best_win