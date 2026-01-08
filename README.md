# 2024~2025 실전문제연구단(X-twice)

스마트폰 내장 초고속 카메라 기반 비접촉식 기계 결함 진단 시스템 

## 1. 프로젝트 개요 
* **기간**: 2024.06 - 2025.01
* **참여자**: 김태완(팀장), 이태훈, 김서랑, 최성현 

---

## 2. 개발 환경 

* **OS**: Ubuntu 22.04 및 Windows 10
* **Language**: Python 3.8+
* **Dependencies**:
* **Hardware**: (예: NVIDIA RTX 3090, STM32F407 Discovery Kit)

---

## 3. 폴더 구조 (예시)
```text
├── src/            # 소스 코드
├── data/           # 데이터셋 
├── docs/           # 관련 문서
├── weights/        # 학습된 모델 가중치 
└── README.md
```

---

## 4. 실행 방법 (예시, Optional)
1. 데이터 전처리: ```python data/preprocess.py```
2. 모델 학습: ```python src/main.py --config config.yaml```
3. 결과 시각화: ```python src/visualize.py```

---

## 5. 브랜치 구조 (예시, Optional)

* **main**: 메인 브랜치
* **feature/xxx**: xxx 기능 개발 브랜치
* **docs**: 문서 관련 브랜치
