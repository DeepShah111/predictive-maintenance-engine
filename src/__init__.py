"""
src package for the AI4I Predictive Maintenance Pipeline.

Importing this package makes the src directory a proper Python package.
All public functions are accessible via their respective modules:

    from src.config             import logger, ARTIFACTS_DIR, ...
    from src.data_ingestion     import load_data, clean_data
    from src.feature_engineering import build_features_and_split, get_preprocessor
    from src.modeling            import train_and_benchmark, tune_champion_model
    from src.evaluation          import evaluate_and_plot, save_model
"""