# RDKit 기초: SMILES부터 Fingerprint까지

## RDKit이란?

**RDKit**은 분자 구조를 다루기 위한 오픈소스 케미인포매틱스(cheminformatics) 툴킷이다. 핵심 엔진은 C++로 작성돼 있고 파이썬 바인딩을 제공해서, 파이썬에서는 pandas나 numpy 쓰듯 `from rdkit import Chem`으로 바로 가져다 쓸 수 있다.

할 수 있는 일을 몇 가지 꼽으면:

- 분자 구조를 여러 포맷(SMILES, SDF, Mol 파일 등)으로 읽고 쓰기
- 구조에서 물리화학적 특성값(서술자, Descriptor) 계산
- 구조를 머신러닝에 넣을 수 있는 숫자 벡터(Fingerprint)로 변환
- 부분구조 검색, 2D/3D 좌표 생성, 화학 반응 처리 등

신약 개발이나 QSAR/QSPR(구조-물성 관계) 모델링에 널리 쓰인다. 이 프로젝트에서는 분자 구조를 머신러닝 입력으로 바꾸는 작업을 모두 RDKit으로 했다.

---

이 프로젝트에서 RDKit을 쓴 흐름은 다음 한 줄로 요약된다.

> **SMILES(문자열) → Mol 객체 → (InChIKey / Fingerprint / Descriptor)**

각 단계에서 쓴 함수와 공식 문서 예제를 순서대로 정리했다.

---

## 1. SMILES와 Mol 객체

**SMILES**(Simplified Molecular Input Line Entry System)는 분자 구조를 한 줄의 문자열로 표현하는 표기법이다. 예를 들어 톨루엔은 `Cc1ccccc1`로 쓴다.

RDKit의 거의 모든 함수는 SMILES 문자열을 직접 받지 않고, 이걸 파싱해서 만든 **Mol 객체**를 입력으로 받는다. 그래서 항상 첫 단계는 `Chem.MolFromSmiles()`다.

```python
>>> from rdkit import Chem
>>> m = Chem.MolFromSmiles('Cc1ccccc1')
>>> m
<rdkit.Chem.rdchem.Mol object at 0x...>
```

### Mol 객체란?

`Chem.MolFromSmiles()`가 반환하는 `rdkit.Chem.rdchem.Mol`은 분자를 **그래프(graph)**로 표현한 RDKit의 핵심 자료구조다. 원자는 노드(node), 결합은 엣지(edge)로 저장되고, 각 원자에는 원소 종류·형식 전하·방향족성 여부·결합된 수소 개수 같은 정보가, 각 결합에는 단일/이중/삼중 결합 여부 같은 정보가 함께 담겨 있다.

SMILES는 텍스트라 원자와 연결 관계를 매번 다시 해석해야 하지만, Mol 객체는 이 정보를 파싱된 구조로 갖고 있다. 그래서 fingerprint, 서술자, 부분구조 검색 같은 작업은 모두 Mol 객체를 대상으로 한다. pandas에서 CSV 파일 대신 `pd.read_csv()`로 읽은 DataFrame으로 작업하는 것과 비슷하다.

필요하면 Mol 객체를 다시 SMILES 문자열로 되돌릴 수도 있다.

```python
>>> from rdkit import Chem
>>> m = Chem.MolFromSmiles('Cc1ccccc1')
>>> Chem.MolToSmiles(m)
'Cc1ccccc1'
```

`MolFromSmiles()`는 파싱에 성공하면 Mol 객체를, 실패하면 `None`을 반환한다. 대량의 SMILES를 다룰 때는 성공률을 먼저 확인했다.

```python
import pandas as pd
from rdkit import Chem

df = pd.DataFrame({"SMILES": ["Cc1ccccc1", "CCO", "not_a_real_smiles"]})

df["mol"] = df["SMILES"].apply(Chem.MolFromSmiles)
success_rate = df["mol"].notna().mean()
print(success_rate)   # 0.666...  (마지막 문자열은 파싱 실패라 None)
```

파싱할 때 RDKit은 sanitization(구조 검증·정규화)도 수행한다. 예를 들어 어떤 원자와도 결합하지 않은 수소가 있으면 `not removing hydrogen atom without neighbors` 경고가 나온다. 파싱이 실패한 건 아니고 이례적인 구조라는 알림이다.

---

## 2. InChI / InChIKey: 구조로 물질 식별하기

서로 다른 데이터셋을 합치려면 같은 물질인지 판단할 기준이 필요하다. 화합물 이름은 동의어와 표기법 차이 때문에 신뢰하기 어려워서 InChIKey를 사용했다.

InChIKey는 IUPAC(국제순수응용화학연맹)과 미국 NIST가 개발한 InChI 표준을 따른다. 그래서 RDKit으로 계산하든 PubChem에서 가져오든 같은 구조면 같은 값이 나오고, 매칭 키로 쓰기에 적합하다.

- `Chem.MolToInchi(mol)`: 표준 InChI 문자열 반환
- `Chem.MolToInchiKey(mol)`: 표준 InChIKey 반환

```python
>>> from rdkit import Chem
>>> mol = Chem.MolFromSmiles('CCO')  # 에탄올
>>> Chem.MolToInchiKey(mol)
'LFQSCWFLJHTTHZ-UHFFFAOYSA-N'
```

InChIKey는 27자 문자열이고 하이픈으로 세 부분이 나뉜다. 앞 14자는 골격(연결성) 정보, 나머지는 입체화학·전하 같은 세부 정보를 반영한다. 그래서 입체이성질체나 염 형태가 다르면 InChIKey도 달라지고, 같은 물질처럼 보여도 매칭되지 않는 경우가 생긴다. 반대로 호변이성질체처럼 SMILES는 다른데 InChIKey가 같게 나오는 경우도 있어서, 병합 후 중복 확인이 필요했다.

---

## 3. Fingerprint: 분자를 벡터로 바꾸기

머신러닝 모델에 넣으려면 분자 구조를 고정 길이 숫자 벡터로 바꿔야 한다. 가장 널리 쓰이는 방법이 Morgan Fingerprint(ECFP, Extended-Connectivity FingerPrint)다.

각 원자를 중심으로 주변 구조(반경 몇 단계 이내)를 순회하면서, 나타나는 국소 패턴들을 해시해서 벡터의 특정 위치를 켜는 방식이다. 구조가 비슷한 분자는 벡터도 비슷하게 나온다.

```python
>>> from rdkit import Chem, DataStructs
>>> from rdkit.Chem import AllChem
>>> fpgen = AllChem.GetMorganGenerator(radius=2)
>>> m1 = Chem.MolFromSmiles('Cc1ccccc1')     # 톨루엔
>>> fp1 = fpgen.GetSparseCountFingerprint(m1)
>>> m2 = Chem.MolFromSmiles('Cc1ncccc1')     # 2-메틸피리딘
>>> fp2 = fpgen.GetSparseCountFingerprint(m2)
>>> DataStructs.DiceSimilarity(fp1, fp2)
0.55...
```

fingerprint를 만드는 함수는 용도에 따라 다르다.

- `GetSparseCountFingerprint()`: 구조 패턴이 나타난 횟수를 세는 희소 벡터. 위 예제처럼 두 분자의 유사도 비교에 쓴다. 길이가 고정되지 않아 `fpSize`가 필요 없다.
- `GetFingerprint()`: 고정 길이 0/1 비트 벡터. 모델 입력은 모든 분자의 길이가 같아야 해서 이 함수를 쓰고, `fpSize`를 지정한다.

```python
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np

mol = Chem.MolFromSmiles('CCO')  # 에탄올
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fp = gen.GetFingerprint(mol)
vector = np.array(fp)   # 길이 2048짜리 0/1 배열
```

- **radius**: 각 원자에서 몇 단계(결합 수)까지 주변 구조를 볼지 결정. 보통 2를 많이 씀
- **fpSize**: 벡터 길이. 클수록 세밀한 구조 차이를 담지만 계산·메모리 비용도 늘어남

---

## 4. Descriptors: 구조에서 계산하는 물성값

fingerprint가 구조 패턴을 벡터로 바꾼 것이라면, 서술자(Descriptor)는 구조에서 계산되는 물리화학적 특성값이다. 실험 없이 구조만으로 계산된다.

```python
>>> from rdkit import Chem
>>> from rdkit.Chem import Descriptors
>>> m = Chem.MolFromSmiles('c1ccccc1C(=O)O')  # 벤조산
>>> Descriptors.TPSA(m)
37.3
>>> Descriptors.MolLogP(m)
1.3848
```

- **TPSA** (Topological Polar Surface Area): 분자 표면 중 극성 원자가 차지하는 면적. 극성/용해도 관련 지표
- **MolLogP**: 옥탄올-물 분배계수. 친유성/친수성 지표

여러 서술자를 한 번에 계산할 때는 `CalcMolDescriptors()`를 쓴다.

```python
>>> from rdkit import Chem
>>> from rdkit.Chem import Descriptors
>>> m = Chem.MolFromSmiles('c1ccccc1C(=O)O')  # 벤조산
>>> vals = Descriptors.CalcMolDescriptors(m)
>>> vals['TPSA']
37.3
>>> vals['NumHDonors']
1
```

RDKit이 지원하는 서술자(이 프로젝트 환경에서는 217개)를 딕셔너리로 반환한다. 이 프로젝트에서는 이 전체를 후보로 두고 permutation importance로 물성별 상위 40개를 골랐다.

---

## 5. 정리

| 단계 | 함수 | 입력 → 출력 | 용도 |
|---|---|---|---|
| 파싱 | `Chem.MolFromSmiles()` | SMILES → Mol 객체 | 모든 작업의 시작점 |
| 식별자 | `Chem.MolToInchiKey()` | Mol → InChIKey(27자) | 데이터셋 간 매칭 키 |
| 지문 | `GetMorganGenerator().GetFingerprint()` | Mol → 고정 길이 비트 벡터 | ML 모델 입력(X) |
| 지문(유사도용) | `GetMorganGenerator().GetSparseCountFingerprint()` | Mol → 희소 카운트 벡터 | 두 분자 간 유사도 비교 |
| 서술자 | `Descriptors.CalcMolDescriptors()` | Mol → 서술자 딕셔너리 | 물성 벡터 보강 |

## 참고자료

- RDKit 공식 문서: https://www.rdkit.org/docs/
- 입문 가이드(Getting Started in Python): https://www.rdkit.org/docs/GettingStartedInPython.html
- API 레퍼런스: https://www.rdkit.org/docs/api-docs.html
