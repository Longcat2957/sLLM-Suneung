#!/bin/bash

# models.json에서 모델 이름 읽기 (jq 필요)
models=$(jq -r 'keys[]' cfg/models.json)
tasks=("kor" "math")

for model in $models; do
    for task in "${tasks[@]}"; do
        echo "===== 모델 ${model} - 태스크 ${task} 테스트 시작 ====="
        python main.py --model "$model" --task "$task"
        if [ $? -ne 0 ]; then
            echo "모델 ${model} - 태스크 ${task} 테스트 실패"
        fi
        echo "===== 모델 ${model} - 태스크 ${task} 테스트 종료 ====="
    done
done
