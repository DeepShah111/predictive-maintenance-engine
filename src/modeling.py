"""
Modeling: Model Zoo, Benchmarking, and Champion Selection.
"""
import pandas as pd
import numpy as np
 
try:
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    logger.warning("CRITICAL: imbalanced-learn not installed. Please pip install imbalanced-learn")
    from sklearn.pipeline import Pipeline
    IMBLEARN_AVAILABLE = False

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Import Config
from src.config import logger, RANDOM_STATE, ARTIFACTS_DIR, COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE

# Advanced Boosting
try:
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    ADVANCED_BOOSTING = True
except ImportError:
    ADVANCED_BOOSTING = False
    logger.warning("   ! XGBoost/CatBoost/LightGBM not found. Skipping advanced models.")

# --- CUSTOM SCORER FOR BUSINESS OPTIMIZATION ---
def total_cost_metric(y_true, y_pred):
    """
    Custom metric to calculate total business cost.
    Goal: MINIMIZE this value.
    """
    cm = confusion_matrix(y_true, y_pred)
    # Handle edge case if model predicts only one class
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        cost = (fp * COST_FALSE_POSITIVE) + (fn * COST_FALSE_NEGATIVE)
        return cost
    return np.inf

# Create scorer: greater_is_better=False because we want lower cost
business_cost_scorer = make_scorer(total_cost_metric, greater_is_better=False)


def train_and_benchmark(X_train, y_train, X_test, y_test, preprocessor):
    logger.info("[4/5] 🏎️ Benchmarking Model Zoo with SMOTE...")
    
    # FIX: Removed scale_weight calculation. 
    # Since SMOTE balances the data to 1:1, we must NOT use class weights.
    
    models = {
        # FIX: Removed class_weight='balanced' from all models
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "SVC (Prob)": SVC(probability=True, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Gaussian NB": GaussianNB()
    }
    
    if ADVANCED_BOOSTING:
        # FIX: Removed scale_pos_weight / class_weight arguments
        models["XGBoost"] = XGBClassifier(
            use_label_encoder=False, 
            eval_metric='logloss', 
            random_state=RANDOM_STATE
        )
        models["CatBoost"] = CatBoostClassifier(
            verbose=0, 
            random_state=RANDOM_STATE
        )
        models["LightGBM"] = LGBMClassifier(
            verbosity=-1, 
            random_state=RANDOM_STATE
        )

    results_list = []
    trained_models = {}

    for name, model in models.items():
        try:
            steps = [('preprocessor', preprocessor)]
            
            if IMBLEARN_AVAILABLE:
                steps.append(('smote', SMOTE(random_state=RANDOM_STATE)))
            
            steps.append(('model', model))
            
            full_pipeline = Pipeline(steps=steps)
            
            # Train
            full_pipeline.fit(X_train, y_train)
            
            # Predict
            y_pred = full_pipeline.predict(X_test)
            try:
                y_prob = full_pipeline.predict_proba(X_test)[:, 1]
            except:
                y_prob = np.zeros(len(y_test))

            # Metrics
            metrics = {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred),
                "F1-Score": f1_score(y_test, y_pred),
                "ROC-AUC": roc_auc_score(y_test, y_prob)
            }
            results_list.append(metrics)
            trained_models[name] = full_pipeline
            
        except Exception as e:
            logger.error(f"   ! Error training {name}: {e}")

    # Leaderboard Generation
    leaderboard = pd.DataFrame(results_list)
    leaderboard = leaderboard.sort_values(by="F1-Score", ascending=False)
    
    leaderboard.to_csv(f"{ARTIFACTS_DIR}/model_leaderboard.csv", index=False)
    
    print("\n" + "="*80 + "\n   🏆 MASTER MODEL LEADERBOARD (Sorted by F1) 🏆\n" + "="*80)
    print(leaderboard.to_string(index=False))
    
    # Select Champion
    champion_name = leaderboard.iloc[0]['Model']
    champion_model = trained_models[champion_name]
    
    logger.info(f"   >>> Champion Selected: {champion_name} (F1: {leaderboard.iloc[0]['F1-Score']:.4f})")
    
    return champion_model, champion_name, leaderboard

def tune_champion_model(model_pipeline, model_name, X_train, y_train):
    logger.info(f"   -> 🔧 Tuning Champion: {model_name}...")
    
    # Tuning Grids
    # FIX: Removed class_weight/scale_pos_weight parameters from grid search 
    # to avoid re-introducing the Double Balancing issue.
    param_grids = {
        "Random Forest": {
            'model__n_estimators': [100, 200],
            'model__max_depth': [10, 20, None]
        },
        "XGBoost": {
            'model__n_estimators': [100, 300],
            'model__learning_rate': [0.01, 0.1],
            'model__max_depth': [3, 6]
        },
        "CatBoost": {
            'model__iterations': [200, 500],
            'model__learning_rate': [0.01, 0.1],
            'model__depth': [4, 6]
        }
    }
    
    grid_params = None
    for key in param_grids:
        if key in model_name:
            grid_params = param_grids[key]
            break
            
    if grid_params:
        # OPTIMIZED: Use Custom Business Cost Scorer
        logger.info("   -> Optimizing for MINIMUM BUSINESS COST (Custom Scorer)...")
        grid = GridSearchCV(
            model_pipeline, 
            grid_params, 
            cv=3, 
            scoring=business_cost_scorer,
            n_jobs=-1, 
            verbose=1
        )
        grid.fit(X_train, y_train)
        logger.info(f"   ✓ Tuning Complete. Best Cost Score: {grid.best_score_:.4f}")
        return grid.best_estimator_
    
    logger.info("   ! No specific tuning grid for this model. Returning base model.")
    return model_pipeline