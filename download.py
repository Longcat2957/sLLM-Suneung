#!/usr/bin/env python3
import os
import sys
import json
import subprocess

# models.json 파일 경로 (필요에 따라 수정)
models_file = os.path.join("cfg", "models.json")

# JSON 파일 로드
try:
    with open(models_file, "r") as f:
        models = json.load(f)
except Exception as e:
    sys.exit(f"모델 파일을 불러오는 데 실패했습니다: {e}")

# 각 모델에 대해 ollama pull 실행
for model_key, model_info in models.items():
    model_name = model_info.get("model")
    if not model_name:
        print(f"'{model_key}' 항목에 model 필드가 없습니다.")
        continue

    print(f"Pulling {model_name} (모델 키: {model_key})...")
    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to pull {model_name}: {e}")
    except Exception as e:
        print(f"Error occurred for {model_name}: {e}")
