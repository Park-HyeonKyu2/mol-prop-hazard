import pandas as pd
import numpy as np
import joblib
import time
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

N_TOP = 40


def compute_all_descriptors(smiles_list):
    desc_names = None
    all_desc = []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        vals = Descriptors.CalcMolDescriptors(m)
        if desc_names is None:
            desc_names = list(vals.keys())
        all_desc.append(list(vals.values()))
    all_desc = np.nan_to_num(np.array(all_desc, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    all_desc = np.clip(all_desc, -1e6, 1e6)
    if "Ipc" in desc_names:
        i = desc_names.index("Ipc")
        desc_names = [d for j, d in enumerate(desc_names) if j != i]
        all_desc = np.delete(all_desc, i, axis=1)
    return desc_names, all_desc


def permutation_importance_train_only(model, X_train, y_train, desc_start, desc_names, n_repeats=3, random_state=42):
    rng = np.random.RandomState(random_state)
    baseline = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    rows = []
    for offset, name in enumerate(desc_names):
        col = desc_start + offset
        drops = []
        for _ in range(n_repeats):
            Xp = X_train.copy()
            rng.shuffle(Xp[:, col])
            score = roc_auc_score(y_train, model.predict_proba(Xp)[:, 1])
            drops.append(baseline - score)
        rows.append({"descriptor": name, "importance": np.mean(drops)})
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


print("\n=== 유해성 12개 태스크 top40 (Ipc 제외, n_estimators=100으로 속도 개선) ===")
t0 = time.time()
tox21 = pd.read_csv("./data/hazard/moleculenet_tox21.csv")
tox21_fp = np.load("./data/processed/moleculenet_tox21_fingerprint.npy")
label_cols = [c for c in tox21.columns if c not in ("SMILES", "InChIKey")]

desc_names, tox21_all_desc = compute_all_descriptors(tox21["SMILES"])
FP_DIM = tox21_fp.shape[1]

all_importance = []
for task in label_cols:
    mask = tox21[task].notna()
    fp_t = tox21_fp[mask.values]
    desc_t = tox21_all_desc[mask.values]
    y_t = tox21.loc[mask, task].values

    X_tr_fp, X_te_fp, desc_tr, desc_te, y_tr, y_te = train_test_split(
        fp_t, desc_t, y_t, test_size=0.2, stratify=y_t, random_state=42
    )
    X_full_tr = np.hstack([X_tr_fp, desc_tr])
    model = RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_full_tr, y_tr)
    imp = permutation_importance_train_only(model, X_full_tr, y_tr, FP_DIM, desc_names)
    imp["task"] = task
    all_importance.append(imp)
    print(f"  {task} 완료 ({time.time()-t0:.0f}초 누적)")

combined = pd.concat(all_importance)
combined["rank"] = combined.groupby("task")["importance"].rank(ascending=False)
avg_rank = combined.groupby("descriptor")["rank"].mean().sort_values()
hazard_top40 = avg_rank.head(N_TOP).index.tolist()

joblib.dump(hazard_top40, "./models/hazard_top40_descriptors.joblib")
combined.to_csv("./models/hazard_descriptor_importance_per_task.csv", index=False)
print("\n유해성 top40 저장 완료:", hazard_top40[:10], "...")
print(f"전체 완료 ({time.time()-t0:.0f}초)")