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
      :root {
        --ink:#2d3b3f; --muted:#6f7774; --copper:#b85c38; --amber:#e7a83e;
        --teal:#138a82; --cream:#fffaf0; --glass:rgba(255,255,255,.78);
      }
      .stApp {
        background:
          radial-gradient(circle at 14% 12%, rgba(241,172,70,.24), transparent 25%),
          radial-gradient(circle at 88% 18%, rgba(31,155,142,.16), transparent 24%),
          linear-gradient(135deg,#fffaf0 0%,#f7f0df 44%,#f5fbf7 100%);
        color:var(--ink);
      }
      .stApp::before {
        content:""; position:fixed; inset:0; pointer-events:none; opacity:.22;
        background-image:linear-gradient(rgba(19,138,130,.14) 1px,transparent 1px),
                         linear-gradient(90deg,rgba(19,138,130,.14) 1px,transparent 1px);
        background-size:42px 42px;
        mask-image:linear-gradient(to bottom,black,transparent 70%);
      }
      [data-testid="stSidebar"] {
        background:linear-gradient(180deg,#fff8e8 0%,#f3ead8 58%,#edf7f2 100%);
        border-right:1px solid rgba(184,92,56,.20);
      }
      [data-testid="stSidebar"] * { color:#33484b !important; }
      [data-testid="stHeader"] { background:rgba(255,250,240,.52); }
      .hero {
        position:relative; overflow:hidden; padding:1.55rem 1.8rem; border-radius:24px;
        background:linear-gradient(118deg,rgba(255,255,255,.92),rgba(255,244,219,.89) 52%,rgba(226,246,240,.88));
        border:1px solid rgba(184,92,56,.20); box-shadow:0 18px 50px rgba(82,66,42,.12);
        margin-bottom:1rem;
      }
      .hero::after {
        content:""; position:absolute; right:-55px; top:-80px; width:260px; height:260px;
        border:28px solid rgba(231,168,62,.14); border-radius:50%;
        box-shadow:0 0 0 24px rgba(19,138,130,.07),0 0 0 52px rgba(184,92,56,.05);
      }
      .hero .eyebrow { color:var(--copper); font-weight:800; letter-spacing:.12em; font-size:.78rem; }
      .hero h1 { margin:.25rem 0 0; color:#29494c; font-size:2.05rem; letter-spacing:.02em; }
      .hero p { margin:.55rem 0 0; color:#65706b; max-width:760px; }
      .step-title { display:flex; align-items:center; gap:.6rem; margin:1.1rem 0 .6rem; }
      .step-badge { display:inline-flex; align-items:center; justify-content:center; width:32px; height:32px;
                    border-radius:10px; color:white; font-weight:800;
                    background:linear-gradient(135deg,var(--copper),var(--amber));
                    box-shadow:0 6px 16px rgba(184,92,56,.22); }
      .step-label { font-size:1.22rem; font-weight:800; color:#334e50; }
      .section-note { color:#707b76; margin:-.2rem 0 .65rem; font-size:.9rem; }
      div[data-testid="stVerticalBlockBorderWrapper"] {
        background:rgba(255,255,255,.68); border-color:rgba(184,92,56,.18) !important;
        box-shadow:0 8px 24px rgba(86,72,48,.06); border-radius:16px;
      }
      div[data-testid="stNumberInput"] input,
      div[data-testid="stTextInput"] input,
      div[data-baseweb="select"] > div {
        background:rgba(255,255,255,.92) !important; border-color:rgba(19,138,130,.24) !important;
      }
      .additive-name { font-weight:800; color:#365355; min-height:1.5rem; }
      .additive-type { color:#9b6b41; font-size:.77rem; margin-bottom:.25rem; }
      .summary-strip { padding:1rem 1.2rem; border-radius:16px; margin:.7rem 0;
                       background:linear-gradient(90deg,rgba(245,182,68,.15),rgba(25,149,136,.11));
                       border:1px solid rgba(19,138,130,.18); color:#3a5354; }
      .result-card { border-radius:18px; padding:1.15rem 1.3rem; color:#30484a;
                     background:rgba(255,255,255,.88); border:1px solid rgba(184,92,56,.18);
                     box-shadow:0 12px 30px rgba(83,65,39,.10); }
      .result-label { font-size:.9rem; color:#687875; }
      .result-value { font-size:2.35rem; font-weight:800; line-height:1.1; margin:.28rem 0; }
      .result-band { font-size:.82rem; color:#71817e; }
      .footnote { color:#7b817b; font-size:.82rem; }
      .stButton>button, .stFormSubmitButton>button {
        border:0; border-radius:12px; font-weight:800; color:white;
        background:linear-gradient(90deg,#b85c38,#e29a38 52%,#138a82);
        box-shadow:0 9px 22px rgba(184,92,56,.18);
      }
      .stButton>button:hover { color:white; border:0; transform:translateY(-1px); }
      div[data-testid="stMetric"] { background:rgba(255,255,255,.63); border:1px solid rgba(19,138,130,.14);
                                    padding:.65rem .8rem; border-radius:14px; }
    </style>
    """,
    unsafe_allow_html=True,
)


DISPLAY_NAMES = {
    "H2SO4": "H₂SO₄", "HCl": "HCl", "K2S2O7": "K₂S₂O₇", "KHSO4": "KHSO₄",
    "FeSO4·7H2O": "FeSO₄·7H₂O", "KOH": "KOH", "CaO": "CaO", "NaCl": "NaCl",
    "CaCl2": "CaCl₂", "SLS(木质素磺酸钠)": "SLS（木质素磺酸钠）", "NaOH": "NaOH",
    "Ca(OH)2": "Ca(OH)₂", "(NH4)2SO4": "(NH₄)₂SO₄", "Na2SO4": "Na₂SO₄",
    "CaSO4": "CaSO₄", "CaCO3": "CaCO₃", "K2SO4": "K₂SO₄", "NaHSO4": "NaHSO₄", "C": "C",
}

ADDITIVE_GROUPS = {
    "酸与酸式盐": {"H2SO4", "HCl", "KHSO4", "NaHSO4"},
    "硫酸盐/焦硫酸盐": {"K2S2O7", "FeSO4·7H2O", "(NH4)2SO4", "Na2SO4", "CaSO4", "K2SO4"},
    "氯化物": {"NaCl", "CaCl2"},
    "碱与钙系": {"KOH", "CaO", "NaOH", "Ca(OH)2", "CaCO3"},
    "辅助剂": {"SLS(木质素磺酸钠)", "C"},
}


@st.cache_resource
def load_assets():
    schema = json.loads((MODEL_DIR / "feature_schema.json").read_text(encoding="utf-8"))
    models = {metal: joblib.load(MODEL_DIR / f"best_{metal}.joblib") for metal in ("Li", "Rb", "Cs")}
    metadata = {
        metal: json.loads((MODEL_DIR / f"metadata_{metal}.json").read_text(encoding="utf-8"))
        for metal in ("Li", "Rb", "Cs")
    }
    return schema, models, metadata


def credentials():
    try:
        auth = st.secrets.get("auth", {})
    except FileNotFoundError:
        auth = {}
    username = auth.get("username") or os.getenv("APP_USERNAME", "")
    password = auth.get("password") or os.getenv("APP_PASSWORD", "")
    return str(username), str(password)


def step_heading(number: int, title: str, note: str = ""):
    st.markdown(
        f'<div class="step-title"><span class="step-badge">{number}</span>'
        f'<span class="step-label">{title}</span></div>', unsafe_allow_html=True,
    )
    if note:
        st.markdown(f'<div class="section-note">{note}</div>', unsafe_allow_html=True)


def additive_group(name: str) -> str:
    return next(group for group, members in ADDITIVE_GROUPS.items() if name in members)


def clear_recipe(additives: list[str]):
    for name in additives:
        st.session_state[f"enabled_{name}"] = False
        st.session_state[f"amount_{name}"] = 0.0


def load_example(additives: list[str]):
    clear_recipe(additives)
    for name, value in {"CaCl2": 0.5, "Na2SO4": 0.4, "C": 0.1}.items():
        st.session_state[f"enabled_{name}"] = True
        st.session_state[f"amount_{name}"] = value


def login_panel():
    expected_user, expected_password = credentials()
    st.markdown(
        '<div class="hero"><div class="eyebrow">AUTUMN · INTELLIGENCE · METALLURGY</div>'
        '<h1>🍂 锂云母智能浸出预测平台</h1>'
        '<p>以数据连接焙烧配方与水浸响应，让每一次工艺筛选更清晰。</p></div>',
        unsafe_allow_html=True,
    )
    left, center, right = st.columns([1.05, 1, 1.05])
    with center:
        with st.container(border=True):
            st.subheader("安全登录")
            user = st.text_input("账号", placeholder="请输入账号")
            password = st.text_input("密码", type="password", placeholder="请输入密码")
            if st.button("进入预测平台", use_container_width=True):
                ok = hmac.compare_digest(user, expected_user) and hmac.compare_digest(password, expected_password)
                if ok:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("账号或密码不正确。")
            st.caption("公开部署时，登录凭据由托管平台的加密变量管理。")


if not st.session_state.get("authenticated", False):
    login_panel()
    st.stop()

schema, models, metadata = load_assets()
ranges = schema["ranges"]
features = schema["feature_names"]
additives = schema["additives"]

with st.sidebar:
    st.markdown("## 🍂 CoreSynergy")
    st.caption("锂云母 Li / Rb / Cs 智能预测")
    st.divider()
    st.markdown("**模型状态**")
    for metal in ("Li", "Rb", "Cs"):
        m = metadata[metal]
        st.markdown(f"**{metal} · {m['best_model']}**")
        st.caption(f"测试集 R² {m['metrics']['测试集R2']:.3f}　RMSE {m['metrics']['测试集RMSE']:.2f}%")
    st.divider()
    st.caption("26项输入全部参与预测；模型用于候选工艺筛选，不替代实验验证。")
    if st.button("退出登录", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

st.markdown(
    '<div class="hero"><div class="eyebrow">CORE SYNERGY · LEPIDOLITE DIGITAL LAB</div>'
    '<h1>锂云母 Li / Rb / Cs 智能浸出预测</h1>'
    '<p>依次选择品位类别、19种焙烧添加剂及用量，再输入焙烧—水浸条件；系统将基于全部26项影响因素输出预测结果。</p></div>',
    unsafe_allow_html=True,
)

tab_predict, tab_variables, tab_about = st.tabs(["🍂 预测工作台", "🧪 全部变量", "📘 模型说明"])

with tab_predict:
    step_heading(1, "原料信息", "当前数据集用0/1表示品位类别，没有矿石具体元素含量与粒度字段。")
    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1, 1.5])
        with c1:
            grade = st.selectbox("品位类别（数据集编码）", [0, 1], help="请按原始数据集中的品位编码选择。")
        with c2:
            st.metric("可用添加剂", f"{len(additives)} 种")
        with c3:
            st.info("配方中未使用的添加剂保持未勾选，其模型输入自动记为0。")

    step_heading(2, "选择焙烧剂并填写用量", "19种添加剂全部来自数据集；用量均为“添加剂与锂云母的质量比”。可选择一种，也可选择多种组成复合配方。")
    for additive in additives:
        st.session_state.setdefault(f"enabled_{additive}", False)
        st.session_state.setdefault(f"amount_{additive}", 0.0)
    b1, b2, space = st.columns([1.15, 1.15, 4])
    with b1:
        if st.button("载入复合配方示例", use_container_width=True):
            load_example(additives)
            st.rerun()
    with b2:
        if st.button("清空全部焙烧剂", use_container_width=True):
            clear_recipe(additives)
            st.rerun()

    additive_values: dict[str, float] = {}
    cols = st.columns(4)
    for idx, name in enumerate(additives):
        r = ranges[name]
        display = DISPLAY_NAMES[name]
        group = additive_group(name)
        with cols[idx % 4]:
            with st.container(border=True):
                st.markdown(f'<div class="additive-name">{idx + 1:02d} · {display}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="additive-type">{group}　数据范围 0–{float(r["max"]):g}</div>', unsafe_allow_html=True)
                enabled = st.checkbox("使用该焙烧剂", key=f"enabled_{name}")
                amount = st.number_input(
                    f"{display} / 锂云母质量比",
                    min_value=float(r["min"]), max_value=float(r["max"]), value=0.0,
                    step=max(0.01, round(float(r["max"]) / 100, 3)),
                    key=f"amount_{name}", disabled=not enabled,
                )
                additive_values[name] = float(amount) if enabled else 0.0

    active = [name for name, value in additive_values.items() if value > 0]
    component_sum = float(sum(additive_values.values()))
    st.markdown(
        f'<div class="summary-strip"><b>当前配方：</b>{" + ".join(DISPLAY_NAMES[n] for n in active) if active else "尚未选择焙烧剂"}'
        f'　　<b>已选：</b>{len(active)}种　　<b>分量合计：</b>{component_sum:.3f}</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        a1, a2 = st.columns([1.15, 1.85])
        with a1:
            use_component_sum = st.toggle("按已选添加剂分量自动计算总质量比", value=True)
        with a2:
            if use_component_sum:
                total_ratio = component_sum
                st.metric("焙烧总添加剂 : 锂云母质量比", f"{total_ratio:.3f}", help="等于所有已选添加剂质量比之和。")
            else:
                total_ratio = st.number_input(
                    "焙烧总添加剂 : 锂云母质量比（手动）",
                    min_value=float(ranges["添加剂总质量比"]["min"]),
                    max_value=float(ranges["添加剂总质量比"]["max"]),
                    value=float(np.clip(component_sum or 1.0, 0, 6)), step=0.05,
                )

    step_heading(3, "输入焙烧与水浸条件", "下列5项均为模型输入；允许范围与训练数据保持一致。")
    process_values: dict[str, float] = {}
    process_labels = ["焙烧温度(℃)", "焙烧时间(h)", "液固比", "水浸温度(℃)", "水浸时间(h)"]
    defaults = [850.0, 1.0, 4.0, 60.0, 1.0]
    process_cols = st.columns(5)
    for col, name, default in zip(process_cols, process_labels, defaults):
        r = ranges[name]
        with col:
            with st.container(border=True):
                process_values[name] = st.number_input(
                    name, min_value=float(r["min"]), max_value=float(r["max"]),
                    value=float(np.clip(default, r["min"], r["max"])),
                    step=1.0 if "温度" in name else 0.1,
                    help=f"训练数据范围：{float(r['min']):g}–{float(r['max']):g}",
                )

    row = {name: 0.0 for name in features}
    row["添加剂总质量比"] = float(total_ratio)
    row["品位"] = float(grade)
    row.update(additive_values)
    row.update({k: float(v) for k, v in process_values.items()})
    input_frame = pd.DataFrame([[row[name] for name in features]], columns=features)

    with st.expander("预测前核对全部26项输入", expanded=False):
        check_rows = [
            {"类别": "原料/总量", "影响因素": "品位类别编码", "输入值": grade},
            {"类别": "原料/总量", "影响因素": "焙烧总添加剂:锂云母质量比", "输入值": round(float(total_ratio), 4)},
        ]
        check_rows += [
            {"类别": additive_group(name), "影响因素": DISPLAY_NAMES[name], "输入值": round(additive_values[name], 4)}
            for name in additives
        ]
        check_rows += [
            {"类别": "焙烧/水浸", "影响因素": name, "输入值": round(value, 4)}
            for name, value in process_values.items()
        ]
        st.dataframe(pd.DataFrame(check_rows), use_container_width=True, hide_index=True, height=520)

    step_heading(4, "计算Li、Rb、Cs预测浸出率")
    submitted = st.button("开始预测三种金属浸出率", type="primary", use_container_width=True)

    if submitted:
        if not active:
            st.warning("当前没有选择焙烧剂。数据中包含少量直接浸出记录，系统仍可计算，但请确认这是否符合实际工艺。")
        if total_ratio > float(ranges["添加剂总质量比"]["max"]):
            st.error("添加剂分量合计已超过数据集总质量比上限6.0，请调整配方后再预测。")
            st.stop()
        if not use_component_sum and abs(float(total_ratio) - component_sum) > 0.15:
            st.warning(f"总质量比（{float(total_ratio):.2f}）与分量之和（{component_sum:.2f}）差异较大，请确认口径。")

        predictions = {}
        result_cols = st.columns(3)
        colors = {"Li": "#138a82", "Rb": "#d98532", "Cs": "#b85c38"}
        for col, metal in zip(result_cols, ("Li", "Rb", "Cs")):
            pred = float(np.clip(models[metal].predict(input_frame)[0], 0, 100))
            rmse = float(metadata[metal]["metrics"]["测试集RMSE"])
            low, high = max(0.0, pred - rmse), min(100.0, pred + rmse)
            predictions[metal] = pred
            with col:
                st.markdown(
                    f'<div class="result-card"><div class="result-label">{metal} 预测浸出率</div>'
                    f'<div class="result-value" style="color:{colors[metal]}">{pred:.1f}%</div>'
                    f'<div class="result-band">参考误差带：{low:.1f}%–{high:.1f}%（±测试集RMSE）</div></div>',
                    unsafe_allow_html=True,
                )
                st.progress(pred / 100)

        st.info("预测结果用于筛选候选条件；正式工艺必须用同一矿样进行平行实验验证。")
        export = input_frame.copy()
        for metal, pred in predictions.items():
            export[f"{metal}预测浸出率(%)"] = round(pred, 3)
        st.download_button(
            "下载本次完整输入与预测结果（CSV）",
            export.to_csv(index=False).encode("utf-8-sig"),
            file_name="lepidolite_prediction.csv", mime="text/csv",
        )

with tab_variables:
    st.subheader("模型使用的全部26项影响因素")
    st.caption("2项原料/总量特征 + 19项添加剂分量 + 5项焙烧—水浸条件。")
    rows = []
    for idx, name in enumerate(features, start=1):
        r = ranges[name]
        if name in additives:
            category = additive_group(name)
            display = DISPLAY_NAMES[name]
            unit = "添加剂/锂云母质量比"
        elif name == "品位":
            category, display, unit = "原料", "品位类别编码", "0或1"
        elif name == "添加剂总质量比":
            category, display, unit = "总量", "焙烧总添加剂:锂云母质量比", "质量比"
        else:
            category, display, unit = "焙烧/水浸", name, "见变量名"
        rows.append({"序号": idx, "类别": category, "影响因素": display, "单位/含义": unit,
                     "数据最小值": r["min"], "数据最大值": r["max"]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=720)

with tab_about:
    st.subheader("模型与适用边界")
    st.markdown(
        """
        - 数据集包含928条Li、557条Rb和542条Cs有效记录，三种金属分别建模。
        - 比较LGBM、随机森林、XGBoost、Stacking、极端随机树、GBDT和SVR共7种模型。
        - 采用固定随机种子492的80/20划分；训练集内部进行5折交叉验证和Optuna-TPE贝叶斯调参。
        - 最佳模型按5折交叉验证RMSE选择：Li为Stacking，Rb与Cs为GBDT。
        - 当前品位字段只有0/1编码；数据集没有粒度、矿石具体元素含量和焙烧气氛。
        """
    )
    summary = pd.DataFrame([
        {"目标": metal, "最佳模型": metadata[metal]["best_model"], "有效样本": metadata[metal]["valid_n"],
         "测试集R²": metadata[metal]["metrics"]["测试集R2"],
         "测试集RMSE": metadata[metal]["metrics"]["测试集RMSE"],
         "五折CV RMSE": metadata[metal]["metrics"]["五折CV_RMSE"]}
        for metal in ("Li", "Rb", "Cs")
    ])
    st.dataframe(
        summary.style.format({"测试集R²": "{:.3f}", "测试集RMSE": "{:.2f}", "五折CV RMSE": "{:.2f}"}),
        use_container_width=True, hide_index=True,
    )
    st.markdown('<p class="footnote">© CoreSynergy · 数据驱动的锂云母工艺筛选辅助工具</p>', unsafe_allow_html=True)
