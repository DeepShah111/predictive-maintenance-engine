"""
Evaluation: Metrics, Confusion Matrix, ROC, and Feature Importance.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from src.config import logger, ARTIFACTS_DIR, COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE

def optimize_threshold(model, X_test, y_test):
    #Finds the optimal probability threshold to minimize business cost.
    y_prob = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0, 1.01, 0.01)
    costs = []
    
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred_t)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            cost = (fp * COST_FALSE_POSITIVE) + (fn * COST_FALSE_NEGATIVE)
        else:
            cost = np.inf 
        costs.append(cost)
        
    min_cost_idx = np.argmin(costs)
    best_threshold = thresholds[min_cost_idx]
    min_cost = costs[min_cost_idx]
    
    logger.info(f"Optimal Threshold Found: {best_threshold:.2f} (Min Cost: ${min_cost:,})")
    return best_threshold

def evaluate_and_plot(model, model_name, X_test, y_test):
    logger.info(f"[5/5] Evaluating & Saving Artifacts for {model_name}")
    
    y_prob = model.predict_proba(X_test)[:, 1]
    
    #OPTIMIZE THRESHOLD
    best_thresh = optimize_threshold(model, X_test, y_test)
    y_pred = (y_prob >= best_thresh).astype(int)
    
    #Classification Report
    print("\n" + "="*30 + f" DETAILED REPORT (Thresh={best_thresh:.2f}) " + "="*30)
    print(classification_report(y_test, y_pred))
    
    #Confusion Matrix & Cost Analysis
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    total_cost = (fp * COST_FALSE_POSITIVE) + (fn * COST_FALSE_NEGATIVE)
    
    print(f"\n BUSINESS IMPACT ANALYSIS:")
    print(f"   - Optimal Threshold: {best_thresh:.2f}")
    print(f"   - False Negatives (Missed Failures): {fn} (Cost: ${fn * COST_FALSE_NEGATIVE:,})")
    print(f"   - False Positives (Wasted Checks):   {fp} (Cost: ${fp * COST_FALSE_POSITIVE:,})")
    print(f"   - TOTAL PROJECTED COST:              ${total_cost:,}")
    print("="*60)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix: {model_name}\nThreshold: {best_thresh:.2f} | Cost: ${total_cost:,}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'{ARTIFACTS_DIR}/graphs/confusion_matrix.png')
    plt.close()
    
    #ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title(f'ROC Curve: {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'{ARTIFACTS_DIR}/graphs/roc_curve.png')
    plt.close()
    
    # Feature Importance
    try:
        classifier = model.named_steps['model']
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        importances = None
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
        elif hasattr(classifier, 'coef_'):
            importances = np.abs(classifier.coef_[0])
            
        if importances is not None:
            indices = np.argsort(importances)[::-1]
            top_n = 10
            
            plt.figure(figsize=(12, 6))
            sns.barplot(x=importances[indices[:top_n]], y=feature_names[indices[:top_n]], palette='viridis')
            plt.title(f'Top {top_n} Feature Importance: {model_name}')
            plt.tight_layout()
            plt.savefig(f'{ARTIFACTS_DIR}/graphs/feature_importance.png')
            plt.close()
        else:
            logger.warning("Model does not support feature importance extraction.")
            
    except Exception as e:
        logger.warning(f"Could not generate feature importance plot: {e}")

def save_model(model, model_name):
    safe_name = model_name.replace(" ", "_").lower()
    path = f'{ARTIFACTS_DIR}/models/{safe_name}_champion.pkl'
    joblib.dump(model, path)
    logger.info(f"Production Model Saved to: {path}")