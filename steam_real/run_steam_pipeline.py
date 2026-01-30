from __future__ import annotations
from src.ingestion import run_ingestion
from src.features import run_feature_pipeline
from src.modeling import run_modeling_pipeline
def main():
    print("=== steam_real pipeline start ===")
    # 1) ingestion
    session_path = run_ingestion()
    # 2) feature building
    ml_path = run_feature_pipeline(session_path)
    # 3) modeling
    _model_path = run_modeling_pipeline(ml_path)
    print("=== steam_real pipeline complete ===")
if __name__ == "__main__":
    main()