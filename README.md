# 분자 구조 기반 화학물질 물성·유해성 예측 및 대체 물질 추천

분자 구조(SMILES) 하나로 **녹는점·끓는점**과 **Tox21 12개 독성 지표**를 예측하고, 구조가 비슷하면서 **더 안전한 대체 물질**을 찾아주는 시스템입니다.

**데모**: https://mol-prop-hazard-noc9kctrcmyysfi7xsnnjc.streamlit.app/

> 무료 호스팅이라 한동안 접속이 없으면 앱이 잠들어 있을 수 있습니다. 처음 열 때 깨어나는 데 잠시 걸릴 수 있어요.

## 배경

화학물질을 바꾸려 할 때는 "물성은 비슷하지만 덜 위험하고 값은 싼 물질"을 찾아야 합니다. 그런데 신규 물질이나 실측값이 없는 물질은 물성과 유해성을 바로 알기 어렵습니다. 기존 그린케미스트리 도구들은 대부분 용매에만 한정되고 가격은 다루지 않아서, 화학물질 전반을 대상으로 물성·유해성·가격을 함께 보는 파이프라인을 만들었습니다.

## 주요 결과

**예측 성능** (RDKit 서술자 217개 중 물성별로 고른 상위 40개를 fingerprint에 추가)

| 대상 | 지표 | Fingerprint만 | + 서술자 40개 | 평가 방식 |
|---|---|---|---|---|
| 녹는점 | R² | 0.647 | **0.762** | 5-fold CV |
| 끓는점 | R² | 0.721 | **0.930** | hold-out |
| 유해성 (12개 태스크) | 평균 ROC-AUC | 0.825 | **0.851** | hold-out |

**최종 모델**

| 대상 | 모델 |
|---|---|
| 녹는점, 끓는점 | HistGradientBoosting (기본 하이퍼파라미터) |
| 유해성 10개 태스크 | RandomForest (`class_weight='balanced'`) |
| 유해성 2개 태스크 (NR-ER-LBD, SR-MMP) | LightGBM (튜닝, CV AUC 0.846 / 0.928) |

**과정에서 확인한 것들**

- 서술자 선정 코드의 첫 버전이 같은 test 데이터로 중요도 계산과 평가를 해서 결과가 부풀려져 있었습니다. 5-fold 교차검증(train으로만 중요도 계산)으로 고치고 elbow curve로 N=40을 정했습니다.
- 녹는점 모델을 30분간 튜닝한 뒤 처음에는 기본값보다 나쁘다고 판단했는데, 두 점수가 서로 다른 교차검증 분할(섞지 않은 5-fold와 섞은 5-fold)에서 나온 값이었습니다. 같은 분할로 다시 비교하니 0.762 vs 0.769로 fold 간 편차(±0.01) 안의 차이여서, 더 단순한 기본값을 유지했습니다.
- HistGradientBoosting과 LightGBM이 소수점 6자리까지 같은 결과를 내서 원인을 분석했습니다. 기본 하이퍼파라미터가 같고, 이진 fingerprint라 분할 지점이 하나뿐이라는 점을 합성 데이터로 재현해 확인했습니다.
- 끓는점 R²가 0.9를 넘었을 때 train/test 간 화합물 중복(InChIKey 기준)을 전수 검사해 누수가 없음을 확인했습니다.

## 추천 방식

1. 입력 물질의 fingerprint(2048비트)와 서술자 6개(분자량, LogP, TPSA, 수소결합 공여체·수용체 수, 회전 가능 결합 수)를 이어붙여 벡터를 만듭니다. 서술자 비중은 화면에서 조절합니다.
2. 14,008개 화합물과 코사인 유사도를 계산합니다.
3. 유해성 12개 태스크 중 어느 것도 입력 물질보다 나쁘지 않고 하나 이상에서 더 낮은 후보(파레토 우위)만 남깁니다. 평균 비교 방식도 선택할 수 있습니다.
4. 녹는점·끓는점 허용 범위, 가격 정보 유무로 추가 필터링할 수 있습니다.

실측값이 있는 물질은 실측값을 쓰고(`measured`), 없는 물질만 모델로 예측합니다(`predicted`).

## 파이프라인

```
CRC Handbook 물성 (녹는점·끓는점)  ─┐
                                     ├─ 정제 (분해·승화·범위값·무기염 제거)
MoleculeNet Tox21 유해성           ─┘        ↓
                                    SMILES → InChIKey (매칭 키)
                                              ↓
                       Morgan Fingerprint + RDKit 서술자 (상위 40개)
                                              ↓
                      물성 회귀 / 유해성 분류 모델 학습 및 선택
                                              ↓
              14,008개 통합 데이터셋 (실측값 + 예측값, measured/predicted 표시)
                                              ↓
                  Streamlit 데모: 유사도 + 유해성 조건으로 대체 물질 추천
```

## 데이터

| 데이터 | 출처 | 규모 |
|---|---|---|
| 물성 (녹는점, 끓는점) | CRC Handbook 기반, [ChemXploreML](https://pubs.acs.org/doi/10.1021/acs.jcim.5c00516) 공개 데이터셋 ([Zenodo](https://zenodo.org/doi/10.5281/zenodo.15007626)) | 7,037 / 4,795개 (정제 후) |
| 유해성 | [MoleculeNet Tox21](https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_Tox21) | 7,831개, 12개 태스크 |
| 가격 | LookChem (robots.txt 허용 범위, 요청 간격 1초 이상, CAPTCHA 발생 시 중단) | 44개 |

원본 데이터는 각 출처의 이용 조건에 따라 저장소에 포함하지 않았습니다. 저장소에는 데모가 사용하는 통합 데이터셋(`data/processed/`)만 들어 있습니다. 가격 데이터는 비상업적 데모 용도로만 사용했습니다.

## 프로젝트 구조

```
├── app.py                  Streamlit 데모
├── requirements.txt        데모 실행용 패키지 (버전 고정)
├── data/processed/         통합 데이터셋 (데모용)
├── models/                 학습된 모델과 서술자 목록 (.joblib)
├── notebooks/
│   ├── 01_property_data_cleaning.ipynb   물성 데이터 정제
│   ├── 02_hazard_data.ipynb              Tox21 데이터 준비
│   ├── 03_fingerprint.ipynb              Morgan fingerprint 생성
│   ├── 04_modeling_experiments.ipynb     모델 비교, 서술자 추가, 서술자 개수 실험
│   ├── 05_price_collection.ipynb         CAS 조회와 가격 수집
│   ├── 06_final_models.ipynb             유해성 LightGBM 튜닝, 최종 모델 조립
│   └── 07_master_dataset.ipynb           14,008개 통합 데이터셋 생성
├── scripts/
│   ├── 01_mp_descriptor_selection_and_tuning.py   녹는점 서술자 선정, 튜닝 vs 기본값 비교
│   ├── 02_descriptor_elbow_bp_hazard.py           끓는점·유해성 elbow curve
│   └── 03_select_top40_bp_hazard.py               끓는점·유해성 상위 40개 선정
└── docs/
    ├── devlog/             개발 기록 (기획부터 배포까지 4편)
    └── notes/              RDKit 정리, HistGB·LightGBM 결과 분석
```

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

노트북과 스크립트는 프로젝트 루트를 작업 디렉터리로 두고 실행했습니다. 원본 데이터와 fingerprint 배열(`.npy`)은 저장소에 없으므로, 재현하려면 위 출처에서 데이터를 받아 `notebooks/01`부터 순서대로 실행해야 합니다. 노트북에는 `xgboost`, `selenium`, `datasets`가 추가로 필요합니다.

## 기술 스택

Python, RDKit, scikit-learn, LightGBM, XGBoost, pandas, NumPy, Streamlit, Selenium

## 한계

- Tox21은 특정 어세이 12개만 다루므로 발암성 등 더 넓은 유해성은 반영하지 않습니다. 유해성 점수가 0이어도 안전하다는 뜻은 아닙니다(예: 벤젠).
- 가격 데이터가 44개뿐이라 비용 비교는 제한적입니다.
- 끓는점·유해성의 서술자 선정은 녹는점과 달리 교차검증 없이 hold-out으로만 확인했습니다.
- 추천은 구조·물성·유해성 유사도만 보며, 실제 구매 가능성이나 규제, 용도 적합성은 고려하지 않습니다.

자세한 내용은 [개발 기록](docs/devlog/)에 정리했습니다.
