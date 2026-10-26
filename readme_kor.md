# ORCHID 한국어 안내

**환자 단위 구강 병리 분류 비교 및 Mamba matched ablation 연구**

[English README](README.md) · [재현 절차](REPRODUCIBILITY.md) · [결과 안내](results/README.md) · [문서 목록](docs/README.md) · [현재 상태](docs/STATUS.md)

BioData Mining 투고 예정 원고를 지원하는 코드와 동결 집계 자료입니다. 현재 **비공개 검토본**이며, 논문 게재나 임상 검증이 완료됐다는 뜻은 아닙니다. 국제 독자에게는 영문 README를 기본 문서로 제공합니다.

코드·집계자료 패키지의 기술 정리 상태와 공개 전 남은 결정은 [최종 인계 안내](docs/STATUS.md)에 모았습니다. 저장소 정리 완료와 환자 연결 자료 공개·전체 실험 재현은 서로 다릅니다.

## 연구 구성

**환자 149명 · 원본 영상 10,072개 · 중복 정리 패치 232,689개**

분류 순서는 Normal=0, OSMF=1, OSCC=2입니다. 패치 확률을 원본 영상 안에서 평균한 뒤, 환자 안의 각 영상을 동일 가중 평균합니다. 환자 단위 outer fold와 별도 inner validation을 사용합니다.

| 구분 | 비교와 run 수 | 체크포인트 선택 | 최대 epoch |
| --- | --- | --- | --- |
| Stage 1 | 네 모델 × 5 folds, seed 42: **20개** | inner-validation 환자 macro OVR AUROC 우선 | 80 |
| Corrected Stage 2 | NoMamba/Mamba × 3 seeds × 5 folds: **30개** | inner-validation 환자 macro F1 우선 | 40 |

Stage 2의 seed 42 Mamba는 **교정된 Stage 2 run**입니다. 과거 Stage 1 Mamba로 대체하지 않습니다. 세 seed는 같은 환자와 분할을 반복하므로 독립 환자 447명이 아닙니다. Stage 1 EfficientNet-Lite2와 Stage 2 Composite-NoMamba도 서로 다른 비교 조건입니다.

## 무엇을 어디서 보나요?

원고 수치와 설명을 대조할 때는 [근거 안내](docs/MANUSCRIPT_EVIDENCE_GUIDE.md)를 먼저 확인하십시오. 출처 파일과 SHA, 집계 자료의 정정 이력을 구분해 제공합니다.

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

최신 검사 수와 검증 범위는 [TEST_REPORT](docs/audits/TEST_REPORT.md)에서 관리합니다. 가상 검사 자료를 연구 관측값으로 해석하면 안 됩니다.

## 재현 범위와 제한

공식 데이터 확보 → 승인된 입력 준비 → 로컬 경로·SHA 연결 → CPU 입력 preflight → 해당 Stage trainer → 평가·figure/table 순서입니다. 실행 명령은 [재현 안내](REPRODUCIBILITY.md), 자료 제공 여부는 [DATA_ACCESS](DATA_ACCESS.md), 과학적·재현성 제한은 [LIMITATIONS](LIMITATIONS.md)에서 각각 관리합니다.

Stage 2는 동일 코호트의 matched ablation이며 독립 외부검증이 아닙니다. 제한 사항은 위 기준 문서에 모아 두었으며, 문서 정리로 미해결 문제가 해결된 것처럼 표현하지 않습니다.

## 라이선스와 공개

권한 있는 자체 소프트웨어 기여에는 [MIT](LICENSE)를 선택했습니다. Codex 보조 사용은 그 자체로 외부 저장소 코드 복사를 뜻하지 않으며, 실제 외부 의존성의 라이선스는 별개로 유지합니다. 출처와 미확인 범위는 [라이선스 안내](docs/licensing/LICENSE_SCOPE.md), [AI 사용 기록](docs/provenance/AI_USE_INVENTORY.md)을 확인하십시오.

자체 소프트웨어에는 MIT를 적용하고 외부 고지는 유지합니다. 승인된 최소 split·patch metadata와 frozen OOF 후보는 [data 안내](data/README.md)에 명시한 별도 범위·출처·라이선스로 포함했습니다. 원본과 연결되는 가명 ID이므로 완전 익명화를 주장하지 않습니다. 저장소는 비공개이며 공개 전환·Release·Zenodo 등록은 하지 않았습니다. 인용 시 정확한 commit과 [CITATION.cff](CITATION.cff)를 확인하십시오.
