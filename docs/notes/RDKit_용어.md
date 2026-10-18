# RDKit 용어·함수 정리

이 프로젝트에서 사용한 RDKit 개념과 함수를 정리했다.

## 1. 핵심 개념

| 용어 | 의미 |
|---|---|
| **SMILES** | 분자 구조를 한 줄 문자열로 표현하는 표기법. 예: 에탄올 = `CCO` |
| **Mol 객체** | RDKit이 SMILES를 파싱해 만드는 분자 표현. fingerprint 생성, InChIKey 계산 등 대부분의 함수가 문자열 대신 Mol 객체를 입력으로 받는다 |
| **Sanitization** | Mol 객체를 만들 때 자동으로 수행되는 구조 검증·정규화. 예를 들어 어떤 원자와도 결합하지 않은 수소가 있으면 `not removing hydrogen atom without neighbors` 경고를 출력한다 |
| **InChI / InChIKey** | IUPAC와 NIST가 만든 구조 기반 식별자 표준. InChIKey는 InChI를 27자 해시로 압축한 것으로, 같은 구조면 어떤 도구로 계산해도 같은 값이 나와 데이터셋 간 매칭 키로 쓴다 |
| **Fingerprint** | 분자 구조를 고정 길이 벡터(주로 0/1 비트)로 바꾼 것. 모델 입력이나 유사도 계산에 쓴다 |
| **Morgan Fingerprint (ECFP)** | 각 원자 주변의 국소 구조를 해시해 비트로 표현하는 방식. 가장 널리 쓰인다 |
| **radius** | Morgan fingerprint에서 각 원자로부터 몇 단계(결합 수)까지 볼지 정하는 값. 이 프로젝트에서는 2 |
| **fpSize** | fingerprint 벡터 길이. 이 프로젝트에서는 2048. 클수록 세밀하지만 계산량이 늘어난다 |
| **Descriptor (서술자)** | 구조에서 바로 계산되는 물리화학적 특성값 (분자량, LogP, TPSA 등) |

## 2. 사용한 함수

| 함수 | 예시 | 역할 |
|---|---|---|
| `Chem.MolFromSmiles(smiles)` | `mol = Chem.MolFromSmiles("CCO")` | SMILES → Mol 객체. 실패하면 `None` |
| `Chem.MolToInchiKey(mol)` | `key = Chem.MolToInchiKey(mol)` | Mol 객체 → InChIKey |
| `rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)` | `gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)` | Morgan fingerprint 생성기. 한 번 만들어 재사용 |
| `gen.GetFingerprint(mol)` | `fp = gen.GetFingerprint(mol)` | fingerprint(비트 벡터) 계산 |
| `np.array(fp)` | `arr = np.array(fp)` | 모델 입력용 numpy 배열로 변환 |
| `Descriptors.CalcMolDescriptors(mol)` | `vals = Descriptors.CalcMolDescriptors(mol)` | 서술자 전체(217개)를 딕셔너리로 계산 |

기본 흐름은 `SMILES` → `MolFromSmiles` → `Mol 객체` → `MolToInchiKey` / `GetFingerprint` / `CalcMolDescriptors`다.

## 3. 자주 쓴 서술자

| 함수 | 의미 |
|---|---|
| `Descriptors.MolWt(mol)` | 분자량 |
| `Descriptors.MolLogP(mol)` | LogP (옥탄올-물 분배계수). 친유성 지표 |
| `Descriptors.TPSA(mol)` | 극성 표면적 |
| `Descriptors.NumHDonors(mol)` / `NumHAcceptors(mol)` | 수소결합 공여체·수용체 수 |
| `Descriptors.NumRotatableBonds(mol)` | 회전 가능한 결합 수 (분자 유연성) |
| `Descriptors.MolMR(mol)` | 몰 굴절률 |

위 6개(MolMR 제외)는 추천 데모의 유사도 계산에 사용했다. 예측 모델에는 217개 중 물성별로 permutation importance로 고른 상위 40개를 사용했다. `Ipc`는 값이 float32 범위를 넘을 수 있어 후보에서 제외했다.

## 4. 공식 문서

- 공식 문서: https://www.rdkit.org/docs/
- Getting Started in Python: https://www.rdkit.org/docs/GettingStartedInPython.html
- API 레퍼런스: https://www.rdkit.org/docs/api-docs.html
