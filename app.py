from __future__ import annotations

import hmac
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"

st.set_page_config(
    page_title="锂云母 Li/Rb/Cs 智能浸出预测",
    page_icon="🍂",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root{
  --pine:#165f52; --teal:#178b82; --amber:#d59a32; --copper:#b9603f;
  --ink:#243b3a; --muted:#6a7976; --paper:#fffdf8;
}
.stApp{
  color:var(--ink);
  background:
    radial-gradient(circle at 7% 5%,rgba(221,154,47,.18),transparent 24%),
    radial-gradient(circle at 94% 8%,rgba(23,139,130,.14),transparent 25%),
    linear-gradient(145deg,#fffefb 0%,#f7f0e2 50%,#eef7f3 100%);
}
.stApp:before{
  content:"";position:fixed;inset:0;pointer-events:none;opacity:.10;
  background-image:linear-gradient(rgba(22,95,82,.13) 1px,transparent 1px),linear-gradient(90deg,rgba(22,95,82,.13) 1px,transparent 1px);
  background-size:48px 48px;mask-image:linear-gradient(to bottom,black,transparent 70%);
}
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#fff9ec,#f3ead8 58%,#eaf5f0);
  border-right:1px solid rgba(185,96,63,.18);
}
[data-testid="stHeader"]{background:rgba(255,253,248,.62)}
.hero{
  position:relative;overflow:hidden;padding:1.55rem 1.85rem;border-radius:24px;
  background:linear-gradient(118deg,rgba(255,255,255,.95),rgba(255,245,221,.92) 54%,rgba(226,246,239,.92));
  border:1px solid rgba(185,96,63,.18);box-shadow:0 18px 50px rgba(78,65,41,.10);margin-bottom:1rem;
}
.hero:after{
  content:"";position:absolute;right:-55px;top:-90px;width:275px;height:275px;border:28px solid rgba(213,154,50,.14);
  border-radius:50%;box-shadow:0 0 0 24px rgba(23,139,130,.07),0 0 0 52px rgba(185,96,63,.05);
}
.eyebrow{color:var(--copper);font-weight:800;letter-spacing:.13em;font-size:.76rem}
.hero h1{margin:.28rem 0 0;color:#244d48;font-size:2.05rem}.hero p{margin:.55rem 0 0;color:#64736f;max-width:870px}
.step{display:flex;align-items:center;gap:.65rem;margin:1.15rem 0 .55rem}
.badge{display:inline-flex;width:32px;height:32px;align-items:center;justify-content:center;border-radius:10px;color:white;font-weight:800;background:linear-gradient(135deg,var(--copper),var(--amber));box-shadow:0 6px 16px rgba(185,96,63,.20)}
.step b{font-size:1.16rem;color:#304f4c}.note{color:#6b7975;margin:-.18rem 0 .65rem;font-size:.90rem}
div[data-testid="stVerticalBlockBorderWrapper"]{background:rgba(255,255,255,.74);border-color:rgba(185,96,63,.16)!important;box-shadow:0 8px 24px rgba(80,67,45,.05);border-radius:16px}
div[data-testid="stNumberInput"] input,div[data-testid="stTextInput"] input,div[data-baseweb="select"]>div{background:rgba(255,255,255,.96)!important;border-color:rgba(23,139,130,.22)!important}
.formula{padding:.8rem 1rem;border-radius:14px;background:linear-gradient(90deg,rgba(213,154,50,.13),rgba(23,139,130,.09));border:1px solid rgba(23,139,130,.16);color:#3d5753}
.addname{font-weight:800;color:#31514e}.addgroup{font-size:.76rem;color:#96683f;margin-bottom:.3rem}
.score{padding:.8rem .85rem;border-radius:14px;background:rgba(255,255,255,.70);border:1px solid rgba(22,95,82,.12);margin:.45rem 0}
.score-title{font-weight:800;color:#274b47}.score-line{font-size:.80rem;color:#667671;margin-top:.18rem}
.result{border-radius:19px;padding:1.15rem 1.30rem;background:rgba(255,255,255,.93);border:1px solid rgba(185,96,63,.17);box-shadow:0 12px 30px rgba(80,64,39,.08)}
.rlabel{font-size:.88rem;color:#697a76}.rvalue{font-size:2.45rem;font-weight:800;line-height:1.08;margin:.28rem 0}.rmeta{font-size:.80rem;color:#6e7e7a;line-height:1.45}
.stButton>button,.stFormSubmitButton>button{border:0;border-radius:12px;font-weight:800;color:white;background:linear-gradient(90deg,#b9603f,#d59a32 52%,#178b82);box-shadow:0 9px 22px rgba(185,96,63,.17)}
.stButton>button:hover{color:white;border:0;transform:translateY(-1px)}
</style>
""",
    unsafe_allow_html=True,
)


DISPLAY = {
    "Li2O_pct": "Li₂O content (%)", "Rb_pct": "Rb content (%)", "Cs_pct": "Cs content (%)",
    "SiO2_pct": "SiO₂ content (%)", "Al2O3_pct": "Al₂O₃ content (%)", "Fe2O3_pct": "Fe₂O₃ content (%)",
    "H2SO4": "H₂SO₄", "HCl": "HCl", "K2S2O7": "K₂S₂O₇", "KHSO4": "KHSO₄",
    "FeSO4_7H2O": "FeSO₄·7H₂O", "KOH": "KOH", "CaO": "CaO", "NaCl": "NaCl",
    "CaCl2": "CaCl₂", "SLS": "SLS（木质素磺酸钠）", "NaOH": "NaOH", "CaOH2": "Ca(OH)₂",
    "NH4_2SO4": "(NH₄)₂SO₄", "Na2SO4": "Na₂SO₄", "CaSO4": "CaSO₄", "CaCO3": "CaCO₃",
    "K2SO4": "K₂SO₄", "NaHSO4": "NaHSO₄", "C": "C",
    "total_additive_ratio": "总添加剂/锂云母质量比",
    "roasting_temp_C": "焙烧温度 (°C)", "roasting_time_h": "焙烧时间 (h)",
    "liquid_solid_ratio": "液固比（液体:固体）", "leaching_temp_C": "水浸温度 (°C)", "leaching_time_h": "水浸时间 (h)",
}

GROUPS = {
    "酸与酸式盐": {"H2SO4", "HCl", "KHSO4", "NaHSO4"},
    "硫酸盐与焦硫酸盐": {"K2S2O7", "FeSO4_7H2O", "NH4_2SO4", "Na2SO4", "CaSO4", "K2SO4"},
    "氯化物": {"NaCl", "CaCl2"},
    "碱与钙系": {"KOH", "CaO", "NaOH", "CaOH2", "CaCO3"},
    "辅助剂": {"SLS", "C"},
}


def feature_engineering(frame: pd.DataFrame, raw_features: list[str]) -> pd.DataFrame:
    x = frame[raw_features].copy()
    additives = [
        "H2SO4", "HCl", "K2S2O7", "KHSO4", "FeSO4_7H2O", "KOH", "CaO", "NaCl",
        "CaCl2", "SLS", "NaOH", "CaOH2", "NH4_2SO4", "Na2SO4", "CaSO4", "CaCO3", "K2SO4", "NaHSO4", "C",
    ]
    eps = 1e-6
    additive_sum = x[additives].fillna(0).sum(axis=1)
    active = x[additives].fillna(0).gt(0)
    x["additive_sum"] = additive_sum
    x["active_additive_count"] = active.sum(axis=1)
    x["is_single_additive"] = active.sum(axis=1).eq(1).astype(float)
    x["is_compound_additive"] = active.sum(axis=1).gt(1).astype(float)
    x["recorded_minus_component_sum"] = x["total_additive_ratio"] - additive_sum
    for additive in additives:
        x[f"frac_{additive}"] = x[additive].fillna(0) / (additive_sum + eps)
    x["sulfate_family"] = x[["H2SO4", "K2S2O7", "KHSO4", "FeSO4_7H2O", "NH4_2SO4", "Na2SO4", "CaSO4", "K2SO4", "NaHSO4"]].fillna(0).sum(axis=1)
    x["chloride_family"] = x[["HCl", "NaCl", "CaCl2"]].fillna(0).sum(axis=1)
    x["alkali_family"] = x[["KOH", "NaOH", "CaOH2"]].fillna(0).sum(axis=1)
    x["calcium_family"] = x[["CaO", "CaCl2", "CaOH2", "CaSO4", "CaCO3"]].fillna(0).sum(axis=1)
    x["acid_family"] = x[["H2SO4", "HCl", "KHSO4", "NaHSO4"]].fillna(0).sum(axis=1)
    x["fe_na_sulfate_synergy"] = x["FeSO4_7H2O"].fillna(0) * x["Na2SO4"].fillna(0)
    x["roasting_severity"] = x["roasting_temp_C"] * np.log1p(x["roasting_time_h"].clip(lower=0))
    x["leaching_severity"] = x["leaching_temp_C"] * np.log1p(x["leaching_time_h"].clip(lower=0))
    x["additive_thermal_load"] = x["total_additive_ratio"] * x["roasting_temp_C"]
    x["liquid_time_exposure"] = x["liquid_solid_ratio"] * x["leaching_time_h"]
    x["Li2O_over_SiO2"] = x["Li2O_pct"] / (x["SiO2_pct"] + eps)
    x["Li2O_over_Al2O3"] = x["Li2O_pct"] / (x["Al2O3_pct"] + eps)
    x["Rb_over_Li2O"] = x["Rb_pct"] / (x["Li2O_pct"] + eps)
    x["Cs_over_Li2O"] = x["Cs_pct"] / (x["Li2O_pct"] + eps)
    x["SiO2_over_Al2O3"] = x["SiO2_pct"] / (x["Al2O3_pct"] + eps)
    return x.replace([np.inf, -np.inf], np.nan)


@st.cache_resource
def load_assets():
    schema = json.loads((MODEL_DIR / "feature_schema.json").read_text(encoding="utf-8"))
    run = json.loads((MODEL_DIR / "run_summary.json").read_text(encoding="utf-8"))
    models = {metal: joblib.load(MODEL_DIR / f"best_{metal}.joblib") for metal in ("Li", "Rb", "Cs")}
    return schema, run, models


def auth_values() -> tuple[str, str]:
    try:
        auth = st.secrets.get("auth", {})
    except FileNotFoundError:
        auth = {}
    return (
        str(auth.get("username") or os.getenv("APP_USERNAME") or "admin"),
        str(auth.get("password") or os.getenv("APP_PASSWORD") or "222333"),
    )


def section(number: int, title: str, note: str = "") -> None:
    st.markdown(f'<div class="step"><span class="badge">{number}</span><b>{title}</b></div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="note">{note}</div>', unsafe_allow_html=True)


def additive_group(name: str) -> str:
    return next(group for group, members in GROUPS.items() if name in members)


def login() -> None:
    expected_user, expected_password = auth_values()
    st.markdown(
        '<div class="hero"><div class="eyebrow">AUTUMN · INTELLIGENCE · GREEN METALLURGY</div>'
        '<h1>🍂 锂云母智能浸出预测平台</h1>'
        '<p>输入矿样组成、全部焙烧添加剂分量及焙烧—水浸条件，一次获得 Li、Rb 和 Cs 浸出率预测。</p></div>',
        unsafe_allow_html=True,
    )
    _, center, _ = st.columns([1.05, 1, 1.05])
    with center, st.container(border=True):
        st.subheader("安全登录")
        username = st.text_input("账号")
        password = st.text_input("密码", type="password")
        if st.button("进入预测平台", width="stretch"):
            if hmac.compare_digest(username, expected_user) and hmac.compare_digest(password, expected_password):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("账号或密码不正确。")
        st.caption("公共演示账号：admin　密码：222333")


if not st.session_state.get("authenticated", False):
    login()
    st.stop()


schema, run, models = load_assets()
features = schema.get("raw_feature_names", schema["feature_names"])
ranges = schema["ranges"]
additives = schema["additive_features"]
composition = schema["composition_features"]

with st.sidebar:
    st.markdown("## 🍂 CoreSynergy")
    st.caption("Lepidolite digital laboratory")
    st.divider()
    st.markdown("### 模型 R²")
    for metal in ("Li", "Rb", "Cs"):
        info = run["best_models"][metal]
        st.markdown(
            f'<div class="score"><div class="score-title">{metal} · {info["best_model"]}</div>'
            f'<div class="score-line">文献留出集 R²　{info["holdout"]["test_R2"]:.3f}</div>'
            f'<div class="score-line">独立实验集 R²　{info["validation46"]["R2"]:.3f}</div></div>',
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption("全部 31 项原始输入均参与预测；模型内部自动构造化学家族、组分占比和工艺强度特征。")
    if st.button("退出登录", width="stretch"):
        st.session_state.authenticated = False
        st.rerun()

st.markdown(
    '<div class="hero"><div class="eyebrow">CORE SYNERGY · DATA-GUIDED PROCESS SCREENING</div>'
    '<h1>Li / Rb / Cs 浸出率预测</h1>'
    '<p>逐项填写矿样化学组成、19 种焙烧添加剂及工艺条件。所有添加剂既可单独使用，也可组合使用。</p></div>',
    unsafe_allow_html=True,
)

prediction_tab, evidence_tab, method_tab = st.tabs(["🍂 预测工作台", "📊 模型 R²", "📘 方法说明"])

with prediction_tab:
    section(1, "输入矿样化学组成", "采用数据集中实际记录的 6 项组成描述符；粒度作为固定前处理条件，不作为模型输入。")
    external_defaults = {"Li2O_pct": 1.98, "Rb_pct": 0.211, "Cs_pct": 0.153, "SiO2_pct": 46.701, "Al2O3_pct": 36.036, "Fe2O3_pct": 3.556}
    composition_values: dict[str, float] = {}
    for start in range(0, len(composition), 3):
        cols = st.columns(3)
        for col, name in zip(cols, composition[start:start + 3]):
            r = ranges[name]
            default = float(external_defaults.get(name, r["median"]))
            upper = max(float(r["max"]) * 1.5, default * 1.25)
            with col, st.container(border=True):
                composition_values[name] = st.number_input(
                    DISPLAY[name], min_value=0.0, max_value=upper, value=default, step=0.01,
                    help=f"文献数据范围：{r['min']:.3g}–{r['max']:.3g}",
                )

    section(2, "选择焙烧添加剂并填写用量", "19 种添加剂全部来自数据集；用量均为该添加剂与锂云母的质量比。")
    if st.button("清空全部添加剂"):
        for additive in additives:
            st.session_state[f"use_{additive}"] = False
            st.session_state[f"dose_{additive}"] = 0.0
        st.rerun()
    additive_values: dict[str, float] = {}
    for start in range(0, len(additives), 4):
        cols = st.columns(4)
        for offset, (col, name) in enumerate(zip(cols, additives[start:start + 4])):
            index = start + offset
            r = ranges[name]
            st.session_state.setdefault(f"use_{name}", False)
            st.session_state.setdefault(f"dose_{name}", 0.0)
            with col, st.container(border=True):
                st.markdown(
                    f'<div class="addname">{index + 1:02d} · {DISPLAY[name]}</div>'
                    f'<div class="addgroup">{additive_group(name)} · 文献上限 {r["max"]:g}</div>',
                    unsafe_allow_html=True,
                )
                enabled = st.checkbox("使用该添加剂", key=f"use_{name}")
                dose = st.number_input(
                    f"{DISPLAY[name]} / 锂云母", min_value=0.0, max_value=float(r["max"]),
                    step=max(0.01, round(float(r["max"]) / 100, 3)), key=f"dose_{name}", disabled=not enabled,
                )
                additive_values[name] = float(dose) if enabled else 0.0

    active = [name for name, value in additive_values.items() if value > 0]
    component_sum = float(sum(additive_values.values()))
    active_text = " + ".join(DISPLAY[name] for name in active) if active else "无添加剂对照"
    st.markdown(
        f'<div class="formula"><b>当前添加剂体系：</b>{active_text}　　<b>分量之和：</b>{component_sum:.3f}</div>',
        unsafe_allow_html=True,
    )
    automatic_total = st.toggle("总添加剂/矿石质量比采用分量之和", value=True)
    if automatic_total:
        total_ratio = component_sum
        st.metric(DISPLAY["total_additive_ratio"], f"{total_ratio:.3f}")
    else:
        total_ratio = st.number_input(
            DISPLAY["total_additive_ratio"], min_value=0.0,
            max_value=max(float(ranges["total_additive_ratio"]["max"]), component_sum),
            value=float(component_sum), step=0.05,
        )

    section(3, "设置焙烧与水浸条件", "液固比按液体:固体输入，例如 10:1 输入 10，0.8:1 输入 0.8。")
    process_names = ["roasting_temp_C", "roasting_time_h", "liquid_solid_ratio", "leaching_temp_C", "leaching_time_h"]
    defaults = [750.0, 1.5, 10.0, 50.0, 1.0]
    process_values: dict[str, float] = {}
    cols = st.columns(5)
    for col, name, default in zip(cols, process_names, defaults):
        r = ranges[name]
        with col, st.container(border=True):
            process_values[name] = st.number_input(
                DISPLAY[name], min_value=float(r["min"]), max_value=float(r["max"]),
                value=float(np.clip(default, r["min"], r["max"])),
                step=1.0 if "temp" in name else 0.1,
                help=f"文献数据范围：{r['min']:g}–{r['max']:g}",
            )

    row = {feature: 0.0 for feature in features}
    row.update(composition_values); row.update(additive_values); row.update(process_values)
    row["total_additive_ratio"] = float(total_ratio)
    raw_frame = pd.DataFrame([[row[feature] for feature in features]], columns=features)
    model_frame = feature_engineering(raw_frame, features)
    expected_model_features = schema.get("model_feature_names")
    if expected_model_features:
        model_frame = model_frame.reindex(columns=expected_model_features)

    section(4, "预测三种金属浸出率")
    if st.button("开始预测 Li、Rb、Cs", type="primary", width="stretch"):
        result_cols = st.columns(3)
        colors = {"Li": "#178b82", "Rb": "#d59a32", "Cs": "#b9603f"}
        export = raw_frame.copy()
        for col, metal in zip(result_cols, ("Li", "Rb", "Cs")):
            prediction = float(np.clip(models[metal].predict(model_frame)[0], 0.0, 100.0))
            info = run["best_models"][metal]
            export[f"{metal}_predicted_recovery_pct"] = prediction
            with col:
                st.markdown(
                    f'<div class="result"><div class="rlabel">{metal} predicted recovery</div>'
                    f'<div class="rvalue" style="color:{colors[metal]}">{prediction:.1f}%</div>'
                    f'<div class="rmeta">{info["best_model"]} · 留出集 R² {info["holdout"]["test_R2"]:.3f}<br>'
                    f'独立实验 R² {info["validation46"]["R2"]:.3f}</div></div>',
                    unsafe_allow_html=True,
                )
                st.progress(prediction / 100)
        st.download_button(
            "下载本次输入与预测结果（CSV）",
            export.to_csv(index=False).encode("utf-8-sig"),
            "lepidolite_prediction.csv",
            "text/csv",
        )

with evidence_tab:
    st.subheader("最佳模型测试集与独立验证集 R²")
    rows = []
    for metal in ("Li", "Rb", "Cs"):
        info = run["best_models"][metal]
        rows.append({
            "Target": metal,
            "Best model": info["best_model"],
            "Literature holdout R²": info["holdout"]["test_R2"],
            "Independent validation R²": info["validation46"]["R2"],
        })
    st.dataframe(
        pd.DataFrame(rows).style.format({"Literature holdout R²": "{:.3f}", "Independent validation R²": "{:.3f}"}),
        hide_index=True, width="stretch",
    )

with method_tab:
    st.subheader("模型与数据说明")
    st.markdown(
        """
- 唯一数据源为合并后的 944 条记录：896 条文献数据用于建模，实验记录仅用于独立验证。
- Li、Rb、Cs 分别采用独立的 XGBoost 模型；训练集内部使用 5 折交叉验证和 Optuna-TPE 贝叶斯优化。
- 46 条主验证样本由响应盲规则确定：完全相同的 31 项输入条件最多保留前三个平行样；完整 48 条结果用于敏感性核验。
- 粒度未作为输入。比较工艺时建议统一控制在 <74 μm，以降低未建模粒度差异。
- 页面显示值限制在物理可解释的 0–100% 区间；平台用于实验条件筛选，最终工艺需以实测结果确认。
"""
    )
