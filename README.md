# 2024~2025 실전문제연구단 (X-twice) 앱 브랜치

스마트폰 내장 초고속 카메라 기반 비접촉식 기계 결함 진단 시스템 

## 1. 프로젝트 개요 
* **기간**: 2024.06 - 2025.01
* **참여자**: 김태완(팀장), 이태훈, 김서랑, 최성현 

---

## 2. 개발 환경 

* **OS**: Ubuntu 22.04, Window 10
* **Language**: kotlin 1.9.0
* **Dependencies**: gradle/libs.versions.toml 참조

---

## 3. 폴더 구조 
```text
├── app/                    # 애플리케이션 메인 모듈
├── OpenCV/                 # OpenCV Android SDK 모듈
├── gradle/
├── gradlew                 # Gradle 실행 래퍼 스크립트 (Linux/macOS용)
├── gradlew.bat             # Gradle 실행 래퍼 스크립트 (Windows용)
├── build.gradle            # 프로젝트 루트 수준 빌드 설정
├── settings.gradle         # 모듈(app, OpenCV) 포함 및 프로젝트 구성 설정
├── gradle.properties       # Gradle 빌드 환경 설정 (JVM Heap, AndroidX 설정 등)
└── local.properties        # 로컬 환경 SDK/NDK 경로 설정 (Git 추적 제외 권장)
```
