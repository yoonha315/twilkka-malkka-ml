import json
import joblib
import os
import xgboost as xgb
from pathlib import Path

def load_model_package(folder_name):
    # 1. Config 먼저 읽기
    config_path = os.path.join(folder_name, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 2. 모델 로드 (방식에 따라 분기)
    if config['save_method'] == "native_json":
        # XGBoost 객체 생성 후 로드
        model = xgb.XGBClassifier()
        model.load_model(os.path.join(folder_name, "model.json"))
    else:
        model = joblib.load(os.path.join(folder_name, "model.joblib"))

    # 3. 스케일러 및 나머지 로드
    scaler_path = os.path.join(folder_name, "scaler.joblib")
    scaler = joblib.load(os.path.join(folder_name, "scaler.joblib")) if Path(scaler_path).exists() else None

    return {
        'model': model,
        'scaler': scaler,
        'threshold': config['threshold'],
        'type': config['model_type']
    }