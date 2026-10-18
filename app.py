# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="화학물질 대체 추천", layout="wide")
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

META_COLS = {"SMILES", "InChIKey", "in_mp", "in_bp", "in_tox21", "mp", "mp_source", "bp", "bp_source"}

# 유사도 계산에만 쓰는 서술자 (예측 모델의 top40과는 별개 -- "실제 대체 가능성"과
# 직결되는 크기/극성/용해도/수소결합 능력 위주로 선정)
SIM_DESCRIPTORS = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds"]


@st.cache_resource
def load_everything():
    master = pd.read_csv("./data/processed/all_compounds_master.csv")
    master = master[master["mp_source"] != "failed"].reset_index(drop=True)

    price = pd.read_csv("./data/processed/all_compounds_with_price.csv")[["InChIKey", "price_usd"]]
    master = master.merge(price, on="InChIKey", how="left")

    hazard_cols = [c for c in master.columns if c not in META_COLS and c != "price_usd" and not c.endswith("_source")]
    master["hazard_score"] = master[hazard_cols].mean(axis=1)

    fps = []
    sim_desc_raw = []
    for smi in master["SMILES"]:
        mol = Chem.MolFromSmiles(smi)
        fps.append(np.array(morgan_gen.GetFingerprint(mol)))
        all_d = Descriptors.CalcMolDescriptors(mol)
        sim_desc_raw.append([all_d.get(name, 0.0) for name in SIM_DESCRIPTORS])
    fps = np.array(fps)
    sim_desc_raw = np.nan_to_num(np.array(sim_desc_raw, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    sim_scaler = StandardScaler().fit(sim_desc_raw)
    sim_desc_scaled = sim_scaler.transform(sim_desc_raw)

    mp_model = joblib.load("./models/mp_model.joblib")
    mp_top40 = joblib.load("./models/top40_descriptors.joblib")
    bp_model = joblib.load("./models/bp_model.joblib")
    bp_top40 = joblib.load("./models/bp_top40_descriptors.joblib")
    tox21_models = joblib.load("./models/tox21_models.joblib")
    hazard_top40 = joblib.load("./models/hazard_top40_descriptors.joblib")

    return (master, hazard_cols, fps, sim_desc_scaled, sim_scaler,
            mp_model, mp_top40, bp_model, bp_top40, tox21_models, hazard_top40)

(master, hazard_cols, fps, sim_desc_scaled, sim_scaler,
 mp_model, mp_top40, bp_model, bp_top40, tox21_models, hazard_top40) = load_everything()


def slice_desc(all_d, names):
    d = np.array([all_d.get(n, 0.0) for n in names], dtype=float)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(d, -1e6, 1e6)


def analyze_query(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    ikey = Chem.MolToInchiKey(mol)
    fp = np.array(morgan_gen.GetFingerprint(mol))
    all_d = Descriptors.CalcMolDescriptors(mol)
    sim_desc_raw = np.array([[all_d.get(name, 0.0) for name in SIM_DESCRIPTORS]], dtype=float)
    sim_desc_raw = np.nan_to_num(sim_desc_raw, nan=0.0, posinf=0.0, neginf=0.0)
    sim_desc = sim_scaler.transform(sim_desc_raw)[0]

    existing = master[master["InChIKey"] == ikey]
    if len(existing) > 0:
        row = existing.iloc[0]
        hazard_probs = {t: row[t] for t in hazard_cols}
        return {
            "fp": fp, "sim_desc": sim_desc, "mp": row["mp"], "mp_source": row["mp_source"],
            "bp": row["bp"], "bp_source": row["bp_source"],
            "hazard_score": row["hazard_score"], "hazard_probs": hazard_probs, "in_master": True,
        }

    mp_pred = mp_model.predict(np.concatenate([fp, slice_desc(all_d, mp_top40)]).reshape(1, -1))[0]
    bp_pred = bp_model.predict(np.concatenate([fp, slice_desc(all_d, bp_top40)]).reshape(1, -1))[0]
    hazard_feat = np.concatenate([fp, slice_desc(all_d, hazard_top40)]).reshape(1, -1)
    hazard_probs = {t: tox21_models[t].predict_proba(hazard_feat)[0, 1] for t in hazard_cols}

    return {
        "fp": fp, "sim_desc": sim_desc, "mp": mp_pred, "mp_source": "predicted",
        "bp": bp_pred, "bp_source": "predicted",
        "hazard_score": np.mean(list(hazard_probs.values())), "hazard_probs": hazard_probs, "in_master": False,
    }


st.title("구조 유사 기반 화학물질 대체 추천")

tab1, tab2 = st.tabs(["SMILES 직접 입력", "목록에서 선택"])
query_smiles = None

def _fill_smiles(value):
    st.session_state["smiles_input"] = value


with tab1:
    smi_input = st.text_input("SMILES 입력", key="smiles_input", placeholder="예: c1ccccc1 (벤젠)")

    if smi_input and len(smi_input) >= 2:
        matches = master[master["SMILES"].str.contains(smi_input, case=False, regex=False, na=False)]
        matches = matches.head(8)
        if len(matches) > 0:
            st.caption("추천 검색어 (클릭하면 입력됩니다)")
            cols = st.columns(4)
            for i, (_, row) in enumerate(matches.iterrows()):
                label = row["SMILES"] if len(row["SMILES"]) <= 18 else row["SMILES"][:18] + "…"
                cols[i % 4].button(
                    label, key=f"suggest_{row['InChIKey']}", use_container_width=True,
                    on_click=_fill_smiles, args=(row["SMILES"],),
                )

    if smi_input:
        query_smiles = smi_input

with tab2:
    options = master["SMILES"] + "  (" + master["InChIKey"].str[:8] + ")"
    choice = st.selectbox("목록에서 선택", options, index=None, placeholder="화합물 검색...")
    if choice:
        query_smiles = choice.split("  (")[0]

ALPHA_OPTIONS = {
    "구조만 보기 (분자 모양만 비교)": 0.0,
    "구조 + 크기·극성·용해도 약간 반영": 0.5,
    "구조 + 크기·극성·용해도 균형있게 반영 (기본)": 1.0,
    "구조 + 크기·극성·용해도 많이 반영": 2.0,
}

col_k, col_alpha = st.columns([1, 3])
with col_k:
    top_k = st.number_input("추천 후보 수", min_value=3, max_value=20, value=10, step=1)
with col_alpha:
    alpha_label = st.selectbox(
        "유사도 판단 기준",
        list(ALPHA_OPTIONS.keys()),
        index=2,
        help=(
            "구조만 보면 겉모양(원자 연결 패턴)이 비슷한 물질을 찾습니다. "
            "물성을 더 반영할수록 분자 크기·극성(TPSA)·용해도(LogP) 같은 실제 특성까지 "
            "비슷한 물질을 우선해서 '진짜 대체 가능한' 후보에 가까워집니다."
        ),
    )
alpha = ALPHA_OPTIONS[alpha_label]

with st.expander("고급 필터 (물성별 개별 조건)", expanded=False):
    col_mp, col_bp = st.columns(2)
    with col_mp:
        mp_filter_on = st.checkbox("녹는점(mp) 범위 제한", value=False)
        mp_tol = st.number_input("mp 허용 오차 (± ºC)", min_value=5, max_value=100, value=30, step=5, disabled=not mp_filter_on)
    with col_bp:
        bp_filter_on = st.checkbox("끓는점(bp) 범위 제한", value=False)
        bp_tol = st.number_input("bp 허용 오차 (± ºC)", min_value=5, max_value=100, value=30, step=5, disabled=not bp_filter_on)

    hazard_task_select = st.multiselect("고려할 유해성 태스크", hazard_cols, default=hazard_cols)
    hazard_mode = st.radio(
        "유해성 조건",
        ["선택한 태스크 중 어느 것도 더 나쁘지 않고, 최소 하나는 더 낮음 (파레토 우위)",
         "선택한 태스크 평균이 더 낮아야 함"],
    )

    price_filter_on = st.checkbox(
        "가격 정보 있는 후보만 보기",
        value=False,
        help="14,008개 화합물 중 44개만 실제 가격(USD)이 확인된 상태입니다.",
    )

if query_smiles:
    result = analyze_query(query_smiles)
    if result is None:
        st.error("SMILES를 해석할 수 없습니다.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("녹는점 예측 (mp)", f"{result['mp']:.1f} ºC", result["mp_source"])
        c2.metric("끓는점 예측 (bp)", f"{result['bp']:.1f} ºC", result["bp_source"])
        c3.metric("유해성 점수 (낮을수록 안전)", f"{result['hazard_score']:.3f}",
                   "DB에 있음" if result["in_master"] else "예측값")

        query_vec = np.concatenate([result["fp"], result["sim_desc"] * alpha]).reshape(1, -1)
        master_vecs = np.hstack([fps, sim_desc_scaled * alpha])
        sims = cosine_similarity(query_vec, master_vecs)[0]
        candidates = master.copy()
        candidates["similarity"] = sims
        candidates = candidates[candidates["similarity"] < 0.999]  # 자기 자신 제외

        if mp_filter_on:
            candidates = candidates[(candidates["mp"] - result["mp"]).abs() <= mp_tol]
        if bp_filter_on:
            candidates = candidates[(candidates["bp"] - result["bp"]).abs() <= bp_tol]

        if hazard_task_select:
            if hazard_mode.startswith("선택한 태스크 중"):
                not_worse = np.ones(len(candidates), dtype=bool)
                better_any = np.zeros(len(candidates), dtype=bool)
                for t in hazard_task_select:
                    not_worse &= (candidates[t] <= result["hazard_probs"][t]).values
                    better_any |= (candidates[t] < result["hazard_probs"][t]).values
                candidates = candidates[not_worse & better_any]
            else:
                query_sub_mean = np.mean([result["hazard_probs"][t] for t in hazard_task_select])
                candidates = candidates[candidates[hazard_task_select].mean(axis=1) < query_sub_mean]

        if price_filter_on:
            candidates = candidates[candidates["price_usd"].notna()]

        candidates = candidates.sort_values("similarity", ascending=False).head(top_k)

        st.subheader(f"더 안전한 대체 후보 ({len(candidates)}개)")
        if len(candidates) == 0:
            st.info("입력 물질보다 유해성이 낮으면서 구조가 유사한 후보를 찾지 못했습니다.")
        else:
            st.dataframe(
                candidates[["SMILES", "similarity", "mp", "bp", "hazard_score", "price_usd"]]
                .rename(columns={"similarity": "구조 유사도", "hazard_score": "유해성 점수", "price_usd": "가격(USD)"})
                .style.format({"구조 유사도": "{:.3f}", "mp": "{:.1f}", "bp": "{:.1f}", "유해성 점수": "{:.3f}", "가격(USD)": "{:.2f}"}, na_rep="-"),
                use_container_width=True,
            )