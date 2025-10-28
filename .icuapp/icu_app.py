from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(page_title="ICU 急変予測", page_icon="🏥", layout="wide")
st.title("🏥 ICU転棟リスク予測（24時間後）")
st.caption("完全ローカル・CSV一括推論。ラベルがあれば評価も可能。")


DEFAULT_COLS_JA = {
    "systolic_bp": "収縮期血圧",
    "diastolic_bp": "拡張期血圧",
    "temperature": "体温",
    "spo2": "SpO2",
    "wbc": "白血球数",
    "neutrophil": "好中球数",
    "age": "年齢",
    "sex": "性別",
    "diagnosis": "診断疾患名",
    "regimen": "レジメン",
    "regimen_days": "レジメン開始後経過日数",
    "timestamp": "測定日時",
    "patient_id": "患者ID",
    "label": "ICU転棟_24h",
}


def detect_encoding(file) -> str:
    # Trial read to decide encoding
    pos = file.tell()
    try:
        pd.read_csv(file)
        enc = "utf-8"
    except UnicodeDecodeError:
        enc = "cp932"
    finally:
        file.seek(pos)
    return enc


def build_pipeline(
    numeric_features: List[str], categorical_features: List[str], model_kind: str
):
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )

    if model_kind == "logreg":
        model = LogisticRegression(max_iter=200, n_jobs=None)
    elif model_kind == "rf":
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        model = LogisticRegression(max_iter=200)

    pipe = Pipeline(
        [
            ("pre", pre),
            ("clf", model),
        ]
    )
    return pipe


def fit_unsupervised_risk(
    df_features: pd.DataFrame,
) -> Tuple[np.ndarray, IsolationForest, MinMaxScaler]:
    # Unsupervised anomaly score scaled to [0,1] as risk
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = imputer.fit_transform(df_features)
    Xs = scaler.fit_transform(X)
    iso = IsolationForest(random_state=42, n_estimators=200, contamination="auto")
    iso.fit(Xs)
    raw = -iso.score_samples(Xs)  # higher => more anomalous
    norm = MinMaxScaler()
    risk = norm.fit_transform(raw.reshape(-1, 1)).reshape(-1)
    return risk, iso, norm


@st.cache_data(show_spinner=False)
def engineer_features(df: pd.DataFrame, colmap: dict) -> pd.DataFrame:
    df = df.copy()
    # Timestamp
    ts_col = colmap.get("timestamp")
    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

    # Basic derived features
    sys_c = colmap.get("systolic_bp")
    dia_c = colmap.get("diastolic_bp")
    neu_c = colmap.get("neutrophil")
    wbc_c = colmap.get("wbc")

    if sys_c in df.columns and dia_c in df.columns:
        df["pulse_pressure"] = df[sys_c] - df[dia_c]

    if neu_c in df.columns and wbc_c in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["nlr"] = df[neu_c] / df[wbc_c].replace(0, np.nan)

    # Per-patient trends if patient_id provided
    pid_c = colmap.get("patient_id")
    if pid_c in df.columns:
        grp = df.sort_values(ts_col).groupby(pid_c)
    else:
        grp = df.sort_values(ts_col).groupby(lambda _: 0)

    for base_col in [
        c
        for c in [
            sys_c,
            dia_c,
            colmap.get("temperature"),
            wbc_c,
            neu_c,
            "pulse_pressure",
            "nlr",
        ]
        if c in df.columns
    ]:
        df[f"{base_col}_diff6h"] = grp[base_col].diff(1)
        df[f"{base_col}_rollmean_12h"] = (
            grp[base_col]
            .rolling(window=2, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    return df


def main():
    st.subheader("CSVアップロード")
    up = st.file_uploader("患者CSVを選択 (6時間ごとの時系列)", type=["csv"])
    if up is None:
        st.info(
            " 列名に含む必要があるもの：『収縮期血圧, 拡張期血圧, 体温, SpO2, 白血球数, 好中球数, 年齢, 性別, 診断疾患名, レジメン, レジメン開始後経過日数, 測定日時』"
        )
        return

    enc = detect_encoding(up)
    df = pd.read_csv(up, encoding=enc)
    st.success(f"CSV読み込み成功（encoding={enc}, 行数={len(df)}）")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("列マッピング")
    cols = list(df.columns)
    colmap = {}
    colmap["patient_id"] = st.selectbox("患者ID (任意)", ["<なし>"] + cols, index=0)
    for k in [
        "systolic_bp",
        "diastolic_bp",
        "temperature",
        "spo2",
        "wbc",
        "neutrophil",
        "age",
        "sex",
        "diagnosis",
        "regimen",
        "regimen_days",
        "timestamp",
    ]:
        default_name = DEFAULT_COLS_JA.get(k)
        options = ["<未指定>"] + cols
        default_idx = options.index(default_name) if default_name in options else 0
        colmap[k] = st.selectbox(f"{k}", options, index=default_idx)

    label_col = st.selectbox(
        "ラベル列（ICU転棟_24h; 0/1, 任意）",
        ["<なし>"] + cols,
        index=(
            (cols.index(DEFAULT_COLS_JA["label"]) + 1)
            if DEFAULT_COLS_JA["label"] in cols
            else 0
        ),
    )
    if colmap["patient_id"] == "<なし>":
        colmap["patient_id"] = None
    # Replace placeholder
    for k, v in list(colmap.items()):
        if v == "<未指定>":
            colmap[k] = None

    df_feat = engineer_features(df, colmap)

    # Choose features
    numeric_candidates = [
        colmap.get("systolic_bp"),
        colmap.get("diastolic_bp"),
        colmap.get("temperature"),
        colmap.get("spo2"),
        colmap.get("wbc"),
        colmap.get("neutrophil"),
        colmap.get("age"),
        colmap.get("regimen_days"),
        "pulse_pressure",
        "nlr",
    ]
    # add engineered numeric deltas
    for base_col in [
        c
        for c in [
            colmap.get("systolic_bp"),
            colmap.get("diastolic_bp"),
            colmap.get("temperature"),
            colmap.get("wbc"),
            colmap.get("neutrophil"),
            "pulse_pressure",
            "nlr",
        ]
        if c is not None
    ]:
        numeric_candidates.extend([f"{base_col}_diff6h", f"{base_col}_rollmean_12h"])
    numeric_features = [c for c in numeric_candidates if c in df_feat.columns]
    categorical_features = [
        c
        for c in [colmap.get("sex"), colmap.get("diagnosis"), colmap.get("regimen")]
        if c in df_feat.columns and c is not None
    ]

    st.write(f"数値特徴量: {numeric_features}")
    st.write(f"カテゴリ特徴量: {categorical_features}")

    st.subheader("モード選択")
    mode = st.radio(
        "推論モード",
        ["教師あり（ラベル必要）", "教師なし（ラベルなしで異常度）"],
        index=0 if label_col != "<なし>" else 1,
    )

    timestamp_col = (
        colmap.get("timestamp") if colmap.get("timestamp") in df_feat.columns else None
    )

    if mode == "教師あり（ラベル必要）" and label_col != "<なし>":
        y = df_feat[label_col].astype(int)
        X = df_feat[numeric_features + categorical_features]

        st.subheader("学習・評価")
        model_kind = st.selectbox(
            "分類器",
            ["logreg", "rf"],
            format_func=lambda x: {
                "logreg": "ロジスティック回帰",
                "rf": "ランダムフォレスト",
            }[x],
        )

        # Time-based split if timestamp available
        if timestamp_col is not None:
            df_ord = df_feat.sort_values(timestamp_col)
            X = df_ord[numeric_features + categorical_features]
            y = df_ord[label_col].astype(int)
        n = len(X)
        split = int(n * 0.75)
        X_tr, y_tr = X.iloc[:split], y.iloc[:split]
        X_te, y_te = X.iloc[split:], y.iloc[split:]

        pipe = build_pipeline(numeric_features, categorical_features, model_kind)
        pipe.fit(X_tr, y_tr)

        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_te)[:, 1]
        else:
            # For models without predict_proba use decision_function if available
            if hasattr(pipe, "decision_function"):
                raw = pipe.decision_function(X_te)
                mm = MinMaxScaler()
                proba = mm.fit_transform(raw.reshape(-1, 1)).reshape(-1)
            else:
                proba = pipe.predict(X_te).astype(float)

        auc = roc_auc_score(y_te, proba) if y_te.nunique() == 2 else np.nan
        ap = average_precision_score(y_te, proba) if y_te.nunique() == 2 else np.nan

        st.write(f"ROC-AUC: {auc:.3f}")
        st.write(f"PR-AUC: {ap:.3f}")

        # Curves
        fpr, tpr, thr = roc_curve(y_te, proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="chance", line=dict(dash="dash")
            )
        )
        fig_roc.update_layout(
            title="ROC", xaxis_title="FPR", yaxis_title="TPR", height=350
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        prec, rec, thr_pr = precision_recall_curve(y_te, proba)
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR"))
        fig_pr.update_layout(
            title="Precision-Recall",
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=350,
        )
        st.plotly_chart(fig_pr, use_container_width=True)

        st.subheader("しきい値・アラート")
        th = st.slider("しきい値", 0.0, 1.0, 0.5, 0.01)
        pred_label = (proba >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, pred_label).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        alert_rate = pred_label.mean() if len(pred_label) else 0.0
        st.write(
            f"感度: {sens:.3f}, 特異度: {spec:.3f}, アラート比率: {alert_rate:.3f}"
        )

        st.subheader("全データで推論・保存")
        # Predict for entire dataset
        if hasattr(pipe, "predict_proba"):
            proba_all = pipe.predict_proba(
                df_feat[numeric_features + categorical_features]
            )[:, 1]
        else:
            raw_all = pipe.decision_function(
                df_feat[numeric_features + categorical_features]
            )
            proba_all = (
                MinMaxScaler()
                .fit_transform(np.array(raw_all).reshape(-1, 1))
                .reshape(-1)
            )
        out = df.copy()
        out["risk_icu24h"] = proba_all
        out["alert"] = (proba_all >= th).astype(int)

        if timestamp_col:
            fig_ts = px.line(
                pd.DataFrame(
                    {"ts": df_feat[timestamp_col], "risk": proba_all}
                ).sort_values("ts"),
                x="ts",
                y="risk",
                title="時系列リスク",
            )
            st.plotly_chart(fig_ts, use_container_width=True)

        csv_bytes = out.to_csv(index=False).encode("utf-8")
        dt = datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button(
            "結果CSVをダウンロード",
            csv_bytes,
            file_name=f"icu24h_predictions_{dt}.csv",
            mime="text/csv",
        )

    else:
        st.subheader("教師なし 異常度ベースのリスク")
        used_cols = [c for c in numeric_features if c is not None]
        if not used_cols:
            st.warning("数値特徴量が選択されていません。最低1列は必要です。")
            return
        risk, iso, norm = fit_unsupervised_risk(df_feat[used_cols])
        out = df.copy()
        out["risk_unsupervised"] = risk
        st.write("各行の異常度を0-1で表示（1に近いほど危険）")

        if timestamp_col:
            fig_ts = px.line(
                pd.DataFrame({"ts": df_feat[timestamp_col], "risk": risk}).sort_values(
                    "ts"
                ),
                x="ts",
                y="risk",
                title="時系列 異常度",
            )
            st.plotly_chart(fig_ts, use_container_width=True)

        th = st.slider("アラートしきい値", 0.0, 1.0, 0.8, 0.01)
        out["alert"] = (risk >= th).astype(int)
        alert_rate = out["alert"].mean()
        st.write(f"アラート比率: {alert_rate:.3f}")

        csv_bytes = out.to_csv(index=False).encode("utf-8")
        dt = datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button(
            "結果CSVをダウンロード",
            csv_bytes,
            file_name=f"icu24h_unsup_{dt}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
