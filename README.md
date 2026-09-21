# Ttareungi Synthetic Detection

따릉이 바구니처럼 복잡하고 변화가 많은 환경에서, AI 기반 합성 이미지 생성으로 이물질 탐지 모델을 학습시키는 방법을 제안한 프로젝트입니다. **MDPI Systems (Volume 14, Issue 5, Article 533)에 게재.**

- **논문**: AI-Based Object Detection in Shared Mobility: A Synthetic Data Approach for Rare Event Recognition
- **진행 기간**: 2025.09 ~ 2026.04
- **진행 형태**: 3인 팀 프로젝트 (연구/논문)
- **기여도**: 제1저자로서 연구 개념화, 합성 데이터 생성 파이프라인 및 비전 모델 개발, 실험 분석, 논문 초안 작성 총괄

이 레포는 본인이 직접 작성한 코드만 정리한 것입니다. 분류 모델(empty/not_empty) 최종 벤치마크 학습은 팀원이 담당했으며, 본 레포에는 포함하지 않았습니다.

## 파이프라인 개요

```
1. roboflow_match.py       Roboflow에서 원본 데이터 다운로드 및 매칭 (match_result_v3.json 생성)
2. meta_cropping.py        원본 YOLOv8 데이터셋에서 바구니 ROI 크롭 + 메타데이터 생성
3. prompt_v1 / v3 / v4.py  Gemini API 기반 합성 이미지 생성 (버전별 프롬프트)
   ├── prompt_v1.py         (논문 V1 — 기본 삽입 프레임워크)
   ├── prompt_v3.py         (논문 V2/V3 — 구조 안정화, damage 파라미터 제거)
   └── prompt_v4.py         (논문 V4 — 최종 채택, 해상도/노이즈 매칭 + 고주파 억제)
4. prepare_batches.py      match_merge가 요구하는 배치 폴더 구조 준비
5. match_merge.py          pHash+SSIM 매칭 후 Poisson Blending으로 scene 수준 재합성
6. basket_labeling.py /
   folder_binary_classify.py   empty/not_empty 라벨링 및 분류
7. total_crops.py, cleanup_duplicate.py,
   download_filter.py, filtering.py,
   generate_labels.py          크롭 통합, 중복 제거, 데이터 누수 방지, YOLO 라벨 변환
8. build_dataset.py        최종 YOLO 탐지 모델 학습셋 구성 (scene-level 재합성 이미지 기반)
```

## 프롬프트 버전 (V1~V4)

프롬프트는 4단계에 걸쳐 점진적으로 개선했습니다. 각 버전은 이전 단계에서 관찰된 실패 사례(비현실적 텍스처, 구조적 왜곡, 과도한 선명도 등)를 보완하는 방향으로 설계했습니다.

| 버전 | 초점 | 핵심 변경 |
|---|---|---|
| V1 | Base Framework | 바구니 구조 보존을 위한 물리적 제약·occlusion 규칙 도입 |
| V2 | Efficiency | `damage` 파라미터를 제거해 프롬프트 복잡도 단순화 |
| V3 | Structural Stability | 동적으로 바뀌던 `content_phrase`를 표준화된 지시문으로 대체해 생성 안정성 강화 |
| V4 | Sim-to-Real Fidelity | 입력 이미지의 해상도·노이즈 레벨에 맞추도록 강제하고, 부자연스러운 고주파 디테일 생성을 억제해 실사 정합성 극대화 |

## YOLO 탐지 모델 학습 설정

비교 모델: YOLOv5n, YOLOv5s, YOLOv7-tiny, YOLOv7-base, YOLOv8n, YOLOv8s (최종 채택: **YOLOv8n**)

| 항목 | 값 |
|---|---|
| 입력 해상도 | 640×640 |
| Epochs | 50 |
| Batch size | 32 |
| Optimizer | auto |

> 학습 실행 코드는 서버 초기화로 소실되어 레포에 포함하지 못했습니다. 위 설정값은 기억을 바탕으로 기록한 것이며, 학습된 가중치(.pt) 자체는 레포에 포함하지 않았습니다.

## 성과 (정량 지표)

| 지표 | Before | After | 변화 |
|---|---|---|---|
| 바구니 검출 mAP@50-95 (YOLOv8n) | 0.695 | 0.736 | +0.041 |
| 바구니 검출 Recall | 0.899 | 0.934 | +0.035 |
| 이물질 분류 F1-Score (MobileNetV3-small) | 0.339 | 0.788 | +44.9%p |

- 합성 이미지 총 3,200장 생성 (4개 프롬프트 버전 × 800장)
- 최종 non-empty 학습 데이터 중 합성 비중 약 71%
- FID 99.71 / KID 0.0104 (V4 프롬프트 기준)

## 참고 — 분류 모델 실험 (본인 개인 실험, 논문 공식 벤치마크 아님)

`reference-experiments/` 폴더의 `mobilenet.py`, `efficientnet.py`는 논문에 실린 공식 분류 모델 벤치마크가 아니라, 본인이 개인적으로 진행한 실험 코드입니다.

- 공통 설정: `IMG_SIZE=224`, `BATCH_SIZE=32`, `EPOCHS=20`, `LEARNING_RATE=0.001`, Adam optimizer, `patience=5`, 2단계 파인튜닝

## 파일 구조

```
ttareungi-synthetic-detection/
├── README.md
├── roboflow_match.py
├── meta_cropping.py
├── prompt_v1.py
├── prompt_v3.py
├── prompt_v4.py
├── prepare_batches.py
├── match_merge.py
├── basket_labeling.py
├── folder_binary_classify.py
├── total_crops.py
├── cleanup_duplicate.py
├── download_filter.py
├── filtering.py
├── generate_labels.py
├── build_dataset.py
└── reference-experiments/
    ├── mobilenet.py
    └── efficientnet.py
```
