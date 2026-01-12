# Open-set Face Identification through Class Incremental Learning

이 프로젝트는 시스템에 등록되지 않은 인물(Open-set)이 등장하고, 새로운 인물이 지속적으로 추가되는(Class Incremental Learning) 실제 환경에서도 안정적으로 동작하는 얼굴 식별 시스템을 연구한 결과물입니다.

## 📌 Project Overview
기존 Softmax 기반 분류 모델은 미등록 인물을 등록된 인물 중 한 명으로 강제 분류하는 한계가 있습니다.
본 프로젝트는 이를 해결하기 위해 **Triplet Loss**를 이용한 임베딩 공간 최적화와 **K-Means 기반 Replay 전략**을 제안합니다.

## 🚀 Key Features
* **Open-set Identification**: Triplet Loss를 통해 동일 인물 간 응집력을 높이고, 거리 기반 식별 거부 메커니즘을 설계하여 미등록 인물 오탐지를 방지합니다.
* **Class Incremental Learning (CIL)**: 새로운 클래스 학습 시 발생하는 기존 지식 망각 문제(Catastrophic Forgetting)를 해결합니다.
* **K-Means Replay Strategy**: 과거 데이터 중 핵심적인 대표 샘플만을 선별하여 메모리 효율을 높이면서도 기존 성능을 안정적으로 유지합니다.
* **Efficient Backbone**: 실시간 추론을 고려하여 연산 효율과 성능의 균형을 맞춘 **MobileNetV3-Large** 모델을 최적의 백본으로 선정하였습니다.

## 📊 Experimental Results
* **Performance Boost**: KFace 데이터셋에서 Softmax 대비 **TPIR 성능을 최대 23.6% 향상**시켰습니다 (FPIR=0.001 기준 97.9%).
* **Knowledge Preservation**: 5단계 증분 학습 후에도 기존 Fine-tuning 방식(86.1%) 대비 압도적으로 높은 **96.6% 이상의 성능을 유지**합니다.

## 🛠️ Usage
1. **Dataset Setup**: KFace 또는 Color FERET 데이터를 준비합니다.
2. **Config**: `config/config.py`에서 `root_path` 및 실험 경로를 설정합니다.
3. **Train**: `python main.py`를 통해 초기 학습 및 증분 학습을 실행합니다.
