"""녹는점(mp) 모델: 서술자 상위 40개 선정 + 튜닝 결과와 기본값 비교 후 저장

프로젝트 루트에서 실행: python scripts/01_mp_descriptor_selection_and_tuning.py
N=40은 notebooks/04_modeling_experiments.ipynb의 5-fold elbow curve로 정했다.
"""
import os
import time

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split

os.makedirs("./models", exist_ok=True)
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# ---------- 1. 상위 40개 서술자 선정 (train으로만 중요도 계산, 누수 방지) ----------
print("=== 1. 상위 40개 서술자 선정 ===")
mp = pd.read_csv("./data/processed/crc_mp_c.csv")
mp_fp = np.load("./data/processed/crc_mp_c_fingerprint.npy")
y_mp = mp["Processed tmp/ºC"].values

desc_names = None
mp_all_desc = []
for smi in mp["SMILES"]:
    vals = Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(smi))
    if desc_names is None:
        desc_names = list(vals.keys())
    mp_all_desc.append(list(vals.values()))
mp_all_desc = np.nan_to_num(np.array(mp_all_desc, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
mp_all_desc = np.clip(mp_all_desc, -1e6, 1e6)

# Ipc: 분자가 커지면 값이 float32 범위를 넘을 수 있어 후보에서 제외
if "Ipc" in desc_names:
    ipc_pos = desc_names.index("Ipc")
    desc_names = [d for i, d in enumerate(desc_names) if i != ipc_pos]
    mp_all_desc = np.delete(mp_all_desc, ipc_pos, axis=1)

X_tr_fp, X_te_fp, desc_tr, desc_te, y_tr, y_te = train_test_split(
    mp_fp, mp_all_desc, y_mp, test_size=0.2, random_state=42
)
X_full_tr = np.hstack([X_tr_fp, desc_tr])
tmp_model = HistGradientBoostingRegressor(random_state=42).fit(X_full_tr, y_tr)

FP_DIM = mp_fp.shape[1]
rng = np.random.RandomState(42)
baseline = r2_score(y_tr, tmp_model.predict(X_full_tr))
rows = []
for offset, name in enumerate(desc_names):
    col = FP_DIM + offset
    drops = []
    for _ in range(3):
        Xp = X_full_tr.copy()
        rng.shuffle(Xp[:, col])
        drops.append(baseline - r2_score(y_tr, tmp_model.predict(Xp)))
    rows.append({"descriptor": name, "importance": np.mean(drops)})
importance_df = pd.DataFrame(rows).sort_values("importance", ascending=False)
TOP40 = importance_df.head(40)["descriptor"].tolist()
joblib.dump(TOP40, "./models/top40_descriptors.joblib")
print("상위 40개 저장 완료")


def descriptor_features(smiles, descriptor_names):
    all_d = Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(smiles))
    d = np.array([all_d.get(name, 0.0) for name in descriptor_names], dtype=float)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(d, -1e6, 1e6)


# ---------- 2. 튜닝 결과와 기본값을 같은 5-fold 분할로 비교 ----------
# 튜닝 개선폭이 fold 간 표준편차보다 클 때만 튜닝 결과를 채택한다.
print("\n=== 2. 녹는점 모델 튜닝 vs 기본값 ===")
t0 = time.time()
X = np.hstack([mp_fp, np.array([descriptor_features(s, TOP40) for s in mp["SMILES"]])])
cv = KFold(n_splits=5, shuffle=True, random_state=42)

default_scores = cross_val_score(HistGradientBoostingRegressor(random_state=42), X, y_mp, cv=cv, scoring="r2")
default_score = default_scores.mean()

reg_param_dist = {
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "max_leaf_nodes": [15, 31, 63, 127],
    "max_iter": [100, 200, 300, 500],
    "min_samples_leaf": [10, 20, 30, 50],
    "l2_regularization": [0.0, 0.1, 1.0],
}
search = RandomizedSearchCV(
    HistGradientBoostingRegressor(random_state=42),
    param_distributions=reg_param_dist,
    n_iter=30, cv=cv, scoring="r2", random_state=42, n_jobs=-1,  # 기본값과 같은 분할 사용
)
search.fit(X, y_mp)

print(f"기본값 R2={default_score:.4f}, 튜닝 R2={search.best_score_:.4f}, {time.time()-t0:.0f}초")
if search.best_score_ - default_score > default_scores.std():
    final_model = search.best_estimator_
    print("튜닝 결과 채택:", search.best_params_)
else:
    final_model = HistGradientBoostingRegressor(random_state=42).fit(X, y_mp)
    print("개선폭이 fold 간 편차 이내라 기본값 채택")

joblib.dump(final_model, "./models/mp_model.joblib")
print("mp_model.joblib 저장 완료")
