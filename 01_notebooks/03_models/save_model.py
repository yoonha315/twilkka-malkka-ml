import joblib
import json
import os

def save_model_package(package_dict, folder_name = 'my_model'):
    os.makedirs(folder_name, exist_ok=True)

    model = package_dict['model']
    model_type = type(model).__name__

    # 모델 저장 (XGBoost만 예외)
    if "XGB" in model_type:
        model_path = os.path.join(folder_name, "model.json")
        model.save_model(model_path)
        save_method = "native_json"
    else:
        model_path = os.path.join(folder_name, "model.joblib")
        joblib.dump(model, model_path)
        save_method = "joblib"

    if package_dict['scaler'] is not None:
        scaler_path = os.path.join(folder_name, "scaler.joblib")
        joblib.dump(package_dict['scaler'], scaler_path)

    # 3. 설정 파일 저장
    config_path = os.path.join(folder_name, "config.json")
    config_data = {
        "model_type": model_type,
        "save_method": save_method,
        "threshold": package_dict['threshold']
    }

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4)

    print(f"✅ [{model_type}] 패키징 완료 ({save_method} 방식)")
