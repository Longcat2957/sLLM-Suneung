# sLLM vs 대학수학능력시험

2024년에 실시된 **2025학년도 대학수학능력시험** 데이터를 이용해 sLLM 간의 성능을 비교하는 코드입니다.  
이 코드는 실제 시험 데이터를 활용하여 sLLM의 문제 해결 능력을 평가하는 데 목적이 있습니다.

## 시스템 환경

해당 코드는 아래와 같은 환경에서 구동하였습니다.

```plaintext
RunPod Pytorch 2.1
18 vCPU 173 GB RAM
1 x H100 NVL
```

## 데이터셋

데이터셋은 2024년에 실시된 **2025학년도 대학수학능력시험** 문제로 구성되어 있으며, 선택 과목은 아래와 같습니다.

> **모든 시험지는 홀수형으로 구성되어 있습니다.**

- **국어영역**: 화법과 작문 선택
- **수학영역**: 미적분 선택
- **영어영역**: 선택과목 없음
- **한국사영역**: 선택과목 없음

## 모델 목록

한국어의 입출력에 최적화(기준: **뇌피셜**)된 모델들을 사용하였습니다.

**뇌피셜** : 대충 LM Studio 넣었을 때 한국어 답변 괜찮게 나오는 애들

| 모델 이름         | 모델 크기 |
|-------------------|-----------|
| gemma2:2b         | 2B        |
| gemma2:9b         | 9B        |
| gemma2:27b        | 27B       |
| phi4              | 14B       |
| qwen2.5:1.5b      | 1.5B      |
| qwen2.5:3b        | 3B        |
| qwen2.5:7b        | 7B        |
| qwen2.5:14b       | 14B       |
| qwen2.5:32b       | 32B       |
| exaone3.5:2.4b    | 2.4B      |
| exaone3.5:7.8b    | 7.8B      |
| exaone3.5:32b     | 32B       |
| mixtral:8x7b      | 8x7B      |
| aya-expanse:8b    | 8B        |
| aya-expanse:32b   | 32B       |

> **참고:** 실제 비교에 사용된 모델의 추가 정보는 추후 업데이트 될 예정입니다.


## requirements.txt

```
vllm

```
> **참고:** ollama 버전은 ollama-version 브랜치를 참고해주십시오

## 성능 평가 코드 예시

TBD

```bash

```



**문의 사항:** 추가적인 정보나 문의가 있으시면 [이메일](mailto:lucete030e@outlook.com)로 연락주시기 바랍니다.