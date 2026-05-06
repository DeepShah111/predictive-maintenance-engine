import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import TARGET_COL, ARTIFACTS_DIR
from src.data_ingestion import load_data, clean_data
from src.feature_engineering import build_features_and_split, get_preprocessor
from src.modeling import train_and_benchmark, tune_champion_model
from src.evaluation import evaluate_and_plot, save_model

print("Starting local pipeline...")

df_raw   = load_data()
df_clean = clean_data(df_raw)
X_train, X_val, X_test, y_train, y_val, y_test = build_features_and_split(df_clean, TARGET_COL)
preprocessor = get_preprocessor()
champion_model, champion_name, leaderboard = train_and_benchmark(X_train, y_train, X_test, y_test, preprocessor)
tuned_champion = tune_champion_model(champion_model, champion_name, X_train, y_train)
evaluate_and_plot(tuned_champion, champion_name, X_val, X_test, y_val, y_test)
save_model(tuned_champion, champion_name)

print(f"\n✅ Model saved to: {ARTIFACTS_DIR / 'models'}")