# ORCHID 비교 연구: 한국어 보조 안내

[English README](README.md)가 논문 심사자와 국제 독자를 위한 기본 문서입니다. 이 파일은 국내 공동연구자 및 저자의 검토를 돕는 보조 안내이며, 새로운 분석 결과나 별도 실행 승인을 제공하지 않습니다.

## 현재 상태

BioData Mining 투고 예정 원고를 지원하는 **비공개 검토용 재현성 후보 패키지**입니다. 논문 게재, 저널 승인, IRB 적합성 또는 코드 공개 승인이 완료됐다는 뜻이 아닙니다. 코드 소유권과 최종 라이선스는 아직 확정하지 않았습니다.

저장소: https://github.com/heedbs/YHY

검토 브랜치: `review/orchid-reproducibility-20261022`. 기본 `main` 브랜치와 구분하십시오. 비공개 링크만 전달하면 심사자가 열람할 수 있는 것은 아니므로 투고 전에 심사 접근 방법 또는 보충자료 제공 방식을 정해야 합니다. 원고에는 최종 검토한 commit을 특정해야 합니다.

## 연구 구성과 결과 해석

- 분류 순서: Normal=0, OSMF=1, OSCC=2.
- 분석 코호트: 환자 149명, source image 10,072개, 중복 정리 patch 232,689개.
- Stage 1: 네 모델, seed 42, 환자 단위 outer 5-fold. 체크포인트는 inner-validation AUROC 우선으로 선택하며 최대 80 epochs입니다. InceptionV3는 physical batch 8 × accumulation 4, 나머지는 16 × 2입니다.
- Corrected Stage 2: Composite-Mamba와 Identity 경로의 Composite-NoMamba를 seeds 42, 20260824, 314159 및 동일한 folds에서 비교합니다. inner-validation 환자 macro F1 우선, 최대 40 epochs입니다. 과거 Stage 1 Mamba seed 42를 corrected Stage 2로 대체하지 않습니다.
- patch 확률을 source image 내부에서 평균한 뒤 환자 내부의 image들을 동일 가중 평균합니다. 반복 seed는 같은 환자와 folds를 사용하므로 독립 환자 447명으로 해석하지 않습니다.
- Stage 2는 동일 코호트의 matched ablation 및 안정성 분석이지 독립 외부검증이 아닙니다. Mamba의 임상적 우월성, 동등성 또는 병변 위치 검증을 주장하지 않습니다.

## 포함 자료

`stage1/`, `stage2/`에는 경로를 이식한 학습·모델·데이터 코드가 있습니다. `configs/`에는 50개 run의 설정 예시와 catalog, `analysis/`에는 평가·통계 코드, `results/`, `table_source_data/`, `figure_source_data/`에는 동결 집계 자료가 있습니다. benchmark, explainability 및 figure 코드는 해당 디렉터리의 reference 자료와 제한 사항을 함께 확인하십시오.

공개 후보에는 원본 병리 이미지, patch cache, checkpoint 및 pretrained weight, 실제 환자별 예측, 식별자 연결표, 정확한 환자 분할 manifest와 내부 감사자료를 넣지 않았습니다. 정확한 분할·manifest 제공은 재현성에 중요하지만 공개 여부가 아직 검토 중이며, 현재 패키지만으로 논문 수치 전체를 처음부터 재현할 수 있다고 주장하지 않습니다.

## 실행과 검증

실제 실행 순서는 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)를 따르십시오. 데이터 확보 → 승인된 manifest 및 검증된 cache 준비 → 새 로컬 경로/config 연결 → 단계별 학습 → 저장된 예측 평가 → figure/table 생성 순서입니다. 문서상 실행 예시와 실제 검증한 명령을 구분해야 합니다.

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python tests/test_cpu.py
sha256sum -c SHA256_MANIFEST.txt
```

위 명령은 적합한 의존성이 설치된 환경을 전제로 합니다. 패키지 제작 당시 CPU 검사 11개를 통과했지만 새 설치 환경 전체, 학습·추론·CUDA 동등성·bootstrap·benchmark·Grad-CAM을 재실행한 것은 아닙니다. synthetic fixture는 연구 환자나 실제 결과가 아닙니다. 과거 실행 환경과 현재 포장 환경은 `environment/`에서 구분합니다.

공개용 코드의 경로 수정 및 Mamba 클래스 추출로 파일 SHA가 달라졌습니다. 일부 과학적 함수와 클래스의 AST 일치 검사는 과거 실행 파일의 바이트 동일성, CUDA 동등성 또는 historical checkpoint의 안전한 resume를 보증하지 않습니다. [CODE_CHANGES.csv](CODE_CHANGES.csv)와 [TEST_REPORT.md](TEST_REPORT.md)를 확인하십시오.

## 데이터·라이선스와 남은 제한

ORCHID 공식 training DOI는 https://doi.org/10.5281/zenodo.12636426 이며 validation/test DOI는 https://doi.org/10.5281/zenodo.12646943 입니다. 확인한 공식 메타데이터의 CC BY 4.0은 데이터 출처에 대한 것으로, 이 저장소 코드 전체의 라이선스나 연구윤리 판단을 대신하지 않습니다. 원래 저장소의 train/val/test 구분과 본 연구의 환자 단위 folds도 별개입니다.

정확한 cache 생성 과정, 일부 과거 환경 버전, 최종 외부 편집 figure의 대응 및 이미지 판독 검증에 미확인 사항이 있습니다. 과거 개발 smoke의 outer 접근 이력도 [LIMITATIONS.md](LIMITATIONS.md)에 공개적으로 기록했습니다. production run의 best 확정 후 outer 평가 1회와 연구 개발 전체 이력은 구분합니다.

투고 전에 코드 권리·라이선스, 자료 제공 방침, 심사 접근 방법, 영문 원고의 Data/Code availability와 저장소 설명의 일치 여부를 확정해야 합니다. 자세한 근거는 [RESULTS_PROVENANCE.md](RESULTS_PROVENANCE.md), [DATA_ACCESS.md](DATA_ACCESS.md), [LICENSE_DECISION_REQUIRED.md](LICENSE_DECISION_REQUIRED.md), [AI_USE_INVENTORY.md](AI_USE_INVENTORY.md)를 확인하십시오.

한국어 설명과 영문 문서가 충돌하면 임의로 해석하지 말고 근거를 확인하여 두 문서를 함께 정정하십시오. 본 문서 추가는 연구 설정이나 결과를 변경하지 않습니다.
