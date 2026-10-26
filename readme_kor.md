# ORCHID 한국어 안내

**환자 단위 구강 병리 분류 비교 및 Mamba matched ablation 연구**

[English README](README.md) · [재현 절차](REPRODUCIBILITY.md) · [결과 안내](results/README.md) · [문서 목록](docs/README.md) · [현재 상태](docs/STATUS.md)

BioData Mining 투고 예정 원고를 지원하는 코드와 동결 집계 자료입니다. 현재 **비공개 검토본**이며, 논문 게재나 임상 검증이 완료됐다는 뜻은 아닙니다. 국제 독자에게는 영문 README를 기본 문서로 제공합니다.

## 연구 구성

**환자 149명 · 원본 영상 10,072개 · 중복 정리 패치 232,689개**

분류 순서는 Normal=0, OSMF=1, OSCC=2입니다. 패치 확률을 원본 영상 안에서 평균한 뒤, 환자 안의 각 영상을 동일 가중 평균합니다. 환자 단위 outer fold와 별도 inner validation을 사용합니다.

| 구분 | 비교와 run 수 | 체크포인트 선택 | 최대 epoch |
| --- | --- | --- | --- |
| Stage 1 | 네 모델 × 5 folds, seed 42: **20개** | inner-validation 환자 macro OVR AUROC 우선 | 80 |
| Corrected Stage 2 | NoMamba/Mamba × 3 seeds × 5 folds: **30개** | inner-validation 환자 macro F1 우선 | 40 |

Stage 2의 seed 42 Mamba는 **교정된 Stage 2 run**입니다. 과거 Stage 1 Mamba로 대체하지 않습니다. 세 seed는 같은 환자와 분할을 반복하므로 독립 환자 447명이 아닙니다. Stage 1 EfficientNet-Lite2와 Stage 2 Composite-NoMamba도 서로 다른 비교 조건입니다.

## 무엇을 어디서 보나요?

| 확인할 내용 | 위치 |
| --- | --- |
| 최종 집계 결과와 표 자료 | [results](results/README.md), [table_source_data](table_source_data/) |
| 모델·학습 코드와 설정 | [Stage 1](stage1/README.md), [Stage 2](stage2/README.md), [configs](configs/README.md) |
| 데이터 확보와 재현 순서 | [DATA_ACCESS](DATA_ACCESS.md), [REPRODUCIBILITY](REPRODUCIBILITY.md) |
| 코드 출처·AI 사용·검사 근거 | [문서 목록](docs/README.md) |
| 미확인 사항과 공개 준비 상태 | [LIMITATIONS](LIMITATIONS.md), [현재 상태](docs/STATUS.md) |

## GPU 없이 확인하기

[CPU 환경 안내](environment/CPU_TEST_ENVIRONMENT.md)에 따라 의존성을 준비한 뒤 저장소 루트에서 실행합니다.

```bash
python -B scripts/check_public_package.py . --manifest
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -B -m unittest discover -s tests -v
```

격리된 CPU 환경에서 **28개 synthetic 검사**를 통과했습니다. 이는 패키지 무결성과 일부 계산 로직의 검사이며, 실제 데이터 학습·추론·CUDA 동등성 검증은 아닙니다. 검사용 가상 자료를 연구 관측값으로 해석하면 안 됩니다.

## 재현 범위와 제한

공식 데이터 확보 → 승인된 정확한 manifest와 검증된 cache 준비 → 로컬 경로·SHA 연결 → 해당 Stage trainer 실행 → 저장된 예측 평가 → figure/table 생성 순서입니다. 자세한 명령은 [재현 안내](REPRODUCIBILITY.md)를 따릅니다.

원본 이미지, cache, checkpoint, 환자별 예측 및 정확한 환자 연결 manifest는 이 패키지에 포함하지 않았습니다. 따라서 현재 저장소만으로 모든 논문 수치를 처음부터 완전 재현할 수 있다고 주장하지 않습니다. cache 최초 생성·과거 환경·최종 외부 편집 figure 대응 등에 남은 제한도 공개적으로 기록했습니다.

Stage 1은 비교 관찰 결과, Stage 2는 동일 코호트의 matched ablation·안정성 분석입니다. 독립 외부검증이 아닙니다. Stage 2의 세 primary CI는 모두 0을 포함하며 우월성이나 동등성을 입증하지 않습니다. InceptionV3의 batch 8 × accumulation 4를 다른 Stage 1 모델의 16 × 2와 혼동하지 마십시오.

## 라이선스와 공개

권한 있는 자체 소프트웨어 기여에는 [MIT](LICENSE)를 선택했습니다. Codex 보조 사용은 그 자체로 외부 저장소 코드 복사를 뜻하지 않으며, 실제 외부 의존성의 라이선스는 별개로 유지합니다. 출처와 미확인 범위는 [라이선스 안내](docs/licensing/LICENSE_SCOPE.md), [AI 사용 기록](docs/provenance/AI_USE_INVENTORY.md)을 확인하십시오.

데이터의 CC BY 4.0을 코드 전체에 적용하거나 연구윤리 판단을 대신하는 근거로 사용하지 않습니다. 이번 문서 정리는 저장소 공개 전환, release 발행, 환자별 자료 공개가 아닙니다. 인용 시 정확한 commit을 특정하고 [CITATION.cff](CITATION.cff)를 확인하십시오.
