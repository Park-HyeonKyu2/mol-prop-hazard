import pandas as pd
import numpy as np
import time
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, roc_auc_score

N_CANDIDATES = [0, 5, 10, 15, 20, 25, 30, 40, 60, 100, 217]


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
    # Ipc 등 일부 서술자가 inf는 아니지만 float32 한계에 근접하는 값을 낼 수 있어 방어적으로 clip
    all_desc = np.clip(all_desc, -1e6, 1e6)
    return desc_names, all_desc


def permutation_importance_train_only(model, X_train, y_train, desc_start, desc_names, metric_fn, n_repeats=3, random_state=42):
    rng = np.random.RandomState(random_state)
    baseline = metric_fn(y_train, model.predict(X_train)) if metric_fn is r2_score else metric_fn(y_train, model.predict_proba(X_train)[:, 1])
    rows = []
    for offset, name in enumerate(desc_names):
        col = desc_start + offset
        drops = []
        for _ in range(n_repeats):
            Xp = X_train.copy()
            rng.shuffle(Xp[:, col])
            score = metric_fn(y_train, model.predict(Xp)) if metric_fn is r2_score else metric_fn(y_train, model.predict_proba(Xp)[:, 1])
            drops.append(baseline - score)
        rows.append({"descriptor": name, "importance": np.mean(drops)})
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


# ============ bp (끓는점) ============
print("=== bp elbow curve ===")
t0 = time.time()
bp = pd.read_csv("./data/processed/crc_bp_c.csv")
bp_fp = np.load("./data/processed/crc_bp_c_fingerprint.npy")
y_bp = bp["Processed tbp/ºC"].values

desc_names, bp_all_desc = compute_all_descriptors(bp["SMILES"])
FP_DIM = bp_fp.shape[1]

X_tr_fp, X_te_fp, desc_tr, desc_te, y_tr, y_te = train_test_split(bp_fp, bp_all_desc, y_bp, test_size=0.2, random_state=42)
X_full_tr = np.hstack([X_tr_fp, desc_tr])
X_full_te = np.hstack([X_te_fp, desc_te])

full_model = HistGradientBoostingRegressor(random_state=42)
full_model.fit(X_full_tr, y_tr)
bp_importance = permutation_importance_train_only(full_model, X_full_tr, y_tr, FP_DIM, desc_names, r2_score)
bp_importance.to_csv("./models/bp_descriptor_importance.csv", index=False)

print(f"{'N':>5} {'R2':>10}")
for n in N_CANDIDATES:
    top_names = bp_importance.head(n)["descriptor"].tolist() if n > 0 else []
    idxs = [desc_names.index(nm) for nm in top_names]
    X_tr = np.hstack([X_tr_fp, desc_tr[:, idxs]]) if n > 0 else X_tr_fp
    X_te = np.hstack([X_te_fp, desc_te[:, idxs]]) if n > 0 else X_te_fp
    m = HistGradientBoostingRegressor(random_state=42)
    m.fit(X_tr, y_tr)
    print(f"{n:>5} {r2_score(y_te, m.predict(X_te)):>10.4f}")
print(f"bp 완료 ({time.time()-t0:.0f}초)\n")

# ============ 유해성 (Tox21, 12개 태스크) ============
print("=== 유해성(Tox21) elbow curve (12개 태스크 평균 AUC) ===")
t0 = time.time()
tox21 = pd.read_csv("./data/hazard/moleculenet_tox21.csv")
tox21_fp = np.load("./data/processed/moleculenet_tox21_fingerprint.npy")
label_cols = [c for c in tox21.columns if c not in ("SMILES", "InChIKey")]

desc_names_tox, tox21_all_desc = compute_all_descriptors(tox21["SMILES"])
FP_DIM = tox21_fp.shape[1]

N_CANDIDATES_CLF = [0, 10, 20, 30, 40, 60, 100, 217]  # 분류 12개라 개수 줄임
auc_by_n = {n: [] for n in N_CANDIDATES_CLF}
per_task_importance = {}

for task in label_cols:
    mask = tox21[task].notna()
    fp_t = tox21_fp[mask.values]
    desc_t = tox21_all_desc[mask.values]
    y_t = tox21.loc[mask, task].values

    X_tr_fp, X_te_fp, desc_tr, desc_te, y_tr, y_te = train_test_split(
        fp_t, desc_t, y_t, test_size=0.2, stratify=y_t, random_state=42
    )
    X_full_tr = np.hstack([X_tr_fp, desc_tr])
    full_model = RandomForestClassifier(class_weight="balanced", n_estimators=300, random_state=42)
    full_model.fit(X_full_tr, y_tr)
    imp = permutation_importance_train_only(full_model, X_full_tr, y_tr, FP_DIM, desc_names_tox, roc_auc_score)
    per_task_importance[task] = imp

    for n in N_CANDIDATES_CLF:
        top_names = imp.head(n)["descriptor"].tolist() if n > 0 else []
        idxs = [desc_names_tox.index(nm) for nm in top_names]
        X_tr = np.hstack([X_tr_fp, desc_tr[:, idxs]]) if n > 0 else X_tr_fp
        X_te = np.hstack([X_te_fp, desc_te[:, idxs]]) if n > 0 else X_te_fp
        m = RandomForestClassifier(class_weight="balanced", n_estimators=300, random_state=42)
        m.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
        auc_by_n[n].append(auc)
    print(f"  {task} 완료 ({time.time()-t0:.0f}초 누적)")

print(f"\n{'N':>5} {'평균 AUC':>10}")
for n in N_CANDIDATES_CLF:
    print(f"{n:>5} {np.mean(auc_by_n[n]):>10.4f}")
print(f"유해성 완료 ({time.time()-t0:.0f}초)")
