# app.py（読み込み部のコア。あなたの既存 app.py に合体してOK）
import json, joblib, pandas as pd, numpy as np, streamlit as st
from datetime import datetime

st.set_page_config(page_title="Titanic SVC", page_icon="🚢", layout="centered")
st.title("🚢 Titanic — SVC Predictor")
st.caption("Notebook からエクスポートした scikit-learn Pipeline を利用")

@st.cache_resource
def load_model_and_meta(model_path="model.pkl", meta_path="model_meta.json"):
    model = joblib.load(model_path)  # 前処理込み Pipeline
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta

model, meta = load_model_and_meta()
feature_cols = meta["feature_cols"]
st.success(f"Model loaded (sklearn {meta['sklearn_version']}).")
st.caption(f"Features: {feature_cols}")

# --- 単発入力（代表的な列のみUI化：足りない列はデフォルト値で補う例） ---
st.subheader("📝 単発入力で予測")
input_vals = {}
for col in feature_cols:
    if col in ("sex","embarked"):
        if col == "sex":
            input_vals[col] = st.selectbox("sex", ["male","female"])
        else:
            input_vals[col] = st.selectbox("embarked", ["S","C","Q"])
    else:
        # 数値列は NumberInput
        input_vals[col] = st.number_input(col, value=0.0, step=1.0)

if st.button("予測する"):
    X_one = pd.DataFrame([input_vals], columns=feature_cols)
    pred   = model.predict(X_one)[0]
    st.write(f"予測クラス（0=死亡, 1=生存）: **{int(pred)}**")
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_one)[0]
        st.write(f"生存確率: **{proba[1]:.3f}**")

# --- CSV 一括推論 ---
st.subheader("📄 CSV アップロードで一括予測（列名は必ずそろえてください）")
csv = st.file_uploader("CSV を選択", type=["csv"])
if csv:
    try:
        df_raw = pd.read_csv(csv)
    except UnicodeDecodeError:
        csv.seek(0)
        df_raw = pd.read_csv(csv, encoding="cp932")

    # 余分な列は無視。不足列は作成（欠損はパイプライン内で補完）
    for c in feature_cols:
        if c not in df_raw.columns:
            df_raw[c] = np.nan
    X = df_raw[feature_cols]

    preds = model.predict(X)
    out = df_raw.copy()
    out["prediction"] = preds
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            out["proba_1"] = proba[:,1]

    dt = datetime.now().strftime("%y%m%d_%H%M")
    st.dataframe(out.head())
    st.download_button("結果をダウンロード", out.to_csv(index=False).encode("utf-8"),
                       file_name=dt+"predictions.csv", mime="text/csv")
