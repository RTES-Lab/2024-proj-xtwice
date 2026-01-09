# 2024~2025 실전문제연구단 (X-twice) 모델 훈련 관련 브랜치

스마트폰 내장 초고속 카메라 기반 비접촉식 기계 결함 진단 시스템 

## 1. 프로젝트 개요 
* **기간**: 2024.06 - 2025.01
* **참여자**: 김태완(팀장), 이태훈, 김서랑, 최성현 

---

## 2. 개발 환경 

* **OS**: Ubuntu 22.04
* **Language**: Python 3.10.12
* **Dependencies**: pip install -r requirements.txt
* **Hardware**: Intel Core i7-13700 CPU @ 5.2 GHz

---

## 3. 폴더 구조 
```text
├── img/ (deprecated)            
├── MDPS-vis-master/    # 변위 추출 
├── model/              # 모델 개발
└── README.md
```

---

## 4. 실행 방법 
1. 변위 추출: ```python MDPS-vis-master/process_video_and_extract_displacement```
2. 모델 학습: ```./model/run_wdcnn.sh```