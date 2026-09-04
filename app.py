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

st.set_page_config(page_title="锂云母 Li/Rb/Cs 智能浸出预测", page_icon="🍂", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root{--pine:#135f4b;--teal:#178b82;--amber:#d59632;--copper:#b65c39;--ink:#24383a;--muted:#667574;}
.stApp{background:radial-gradient(circle at 8% 6%,rgba(222,155,51,.20),transparent 23%),radial-gradient(circle at 92% 8%,rgba(23,139,130,.16),transparent 24%),linear-gradient(145deg,#fffdf8 0%,#f7f1e4 48%,#edf7f3 100%);color:var(--ink)}
.stApp:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.16;background-image:linear-gradient(rgba(19,95,75,.12) 1px,transparent 1px),linear-gradient(90deg,rgba(19,95,75,.12) 1px,transparent 1px);background-size:46px 46px;mask-image:linear-gradient(to bottom,black,transparent 72%)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#fff9ed,#f2ead8 58%,#eaf4ef);border-right:1px solid rgba(182,92,57,.18)}
[data-testid="stHeader"]{background:rgba(255,253,248,.65)}
.hero{position:relative;overflow:hidden;padding:1.55rem 1.8rem;border-radius:24px;background:linear-gradient(118deg,rgba(255,255,255,.94),rgba(255,245,221,.92) 53%,rgba(226,246,239,.91));border:1px solid rgba(182,92,57,.19);box-shadow:0 18px 50px rgba(78,65,41,.11);margin-bottom:1rem}
.hero:after{content:"";position:absolute;right:-58px;top:-86px;width:270px;height:270px;border:28px solid rgba(213,150,50,.14);border-radius:50%;box-shadow:0 0 0 24px rgba(23,139,130,.07),0 0 0 52px rgba(182,92,57,.05)}
.eyebrow{color:var(--copper);font-weight:800;letter-spacing:.12em;font-size:.78rem}.hero h1{margin:.25rem 0 0;color:#244c48;font-size:2.05rem}.hero p{margin:.55rem 0 0;color:#63716d;max-width:820px}
.step{display:flex;align-items:center;gap:.62rem;margin:1.15rem 0 .55rem}.badge{display:inline-flex;width:32px;height:32px;align-items:center;justify-content:center;border-radius:10px;color:white;font-weight:800;background:linear-gradient(135deg,var(--copper),var(--amber));box-shadow:0 6px 16px rgba(182,92,57,.2)}.step b{font-size:1.18rem;color:#304f4e}.note{color:#6c7975;margin:-.2rem 0 .65rem;font-size:.9rem}
div[data-testid="stVerticalBlockBorderWrapper"]{background:rgba(255,255,255,.72);border-color:rgba(182,92,57,.16)!important;box-shadow:0 8px 24px rgba(80,67,45,.055);border-radius:16px}
div[data-testid="stNumberInput"] input,div[data-testid="stTextInput"] input,div[data-baseweb="select"]>div{background:rgba(255,255,255,.95)!important;border-color:rgba(23,139,130,.22)!important}
.formula{padding:.8rem 1rem;border-radius:14px;background:linear-gradient(90deg,rgba(213,150,50,.13),rgba(23,139,130,.09));border:1px solid rgba(23,139,130,.16);color:#3c5552}.addname{font-weight:800;color:#31514f}.addgroup{font-size:.76rem;color:#9a6940;margin-bottom:.3rem}
.result{border-radius:18px;padding:1.1rem 1.25rem;background:rgba(255,255,255,.9);border:1px solid rgba(182,92,57,.17);box-shadow:0 12px 30px rgba(80,64,39,.09)}.rlabel{font-size:.88rem;color:#697a76}.rvalue{font-size:2.35rem;font-weight:800;line-height:1.1;margin:.28rem 0}.rmeta{font-size:.8rem;color:#6e7e7a}
.stButton>button,.stFormSubmitButton>button{border:0;border-radius:12px;font-weight:800;color:white;background:linear-gradient(90deg,#b65c39,#d59632 52%,#178b82);box-shadow:0 9px 22px rgba(182,92,57,.17)}.stButton>button:hover{color:white;border:0;transform:translateY(-1px)}
</style>
""", unsafe_allow_html=True)

DISPLAY = {
"Li2O_pct":"Li₂O content (%)","Rb_pct":"Rb content (%)","Cs_pct":"Cs content (%)","SiO2_pct":"SiO₂ content (%)","Al2O3_pct":"Al₂O₃ content (%)","Fe2O3_pct":"Fe₂O₃ content (%)",
"H2SO4":"H₂SO₄","HCl":"HCl","K2S2O7":"K₂S₂O₇","KHSO4":"KHSO₄","FeSO4_7H2O":"FeSO₄·7H₂O","KOH":"KOH","CaO":"CaO","NaCl":"NaCl","CaCl2":"CaCl₂","SLS":"SLS（木质素磺酸钠）","NaOH":"NaOH","CaOH2":"Ca(OH)₂","NH4_2SO4":"(NH₄)₂SO₄","Na2SO4":"Na₂SO₄","CaSO4":"CaSO₄","CaCO3":"CaCO₃","K2SO4":"K₂SO₄","NaHSO4":"NaHSO₄","C":"C",
"roasting_temp_C":"焙烧温度 (°C)","roasting_time_h":"焙烧时间 (h)","liquid_solid_ratio":"液固比（数值）","leaching_temp_C":"水浸温度 (°C)","leaching_time_h":"水浸时间 (h)","total_additive_ratio":"总添加剂/锂云母质量比"
}
GROUPS={"酸与酸式盐":{"H2SO4","HCl","KHSO4","NaHSO4"},"硫酸盐/焦硫酸盐":{"K2S2O7","FeSO4_7H2O","NH4_2SO4","Na2SO4","CaSO4","K2SO4"},"氯化物":{"NaCl","CaCl2"},"碱与钙系":{"KOH","CaO","NaOH","CaOH2","CaCO3"},"辅助剂":{"SLS","C"}}

@st.cache_resource
def load_assets():
    schema=json.loads((MODEL_DIR/"feature_schema.json").read_text(encoding="utf-8"))
    run=json.loads((MODEL_DIR/"run_summary.json").read_text(encoding="utf-8"))
    models={m:joblib.load(MODEL_DIR/f"best_{m}.joblib") for m in ("Li","Rb","Cs")}
    return schema,run,models

def auth_values():
    try: auth=st.secrets.get("auth",{})
    except FileNotFoundError: auth={}
    return str(auth.get("username") or os.getenv("APP_USERNAME") or "admin"),str(auth.get("password") or os.getenv("APP_PASSWORD") or "222333")

def heading(n,title,note=""):
    st.markdown(f'<div class="step"><span class="badge">{n}</span><b>{title}</b></div>',unsafe_allow_html=True)
    if note: st.markdown(f'<div class="note">{note}</div>',unsafe_allow_html=True)

def group_of(name): return next(k for k,v in GROUPS.items() if name in v)

def login():
    expected_user,expected_password=auth_values()
    st.markdown('<div class="hero"><div class="eyebrow">AUTUMN · INTELLIGENCE · GREEN METALLURGY</div><h1>🍂 锂云母智能浸出预测平台</h1><p>用矿样组成与焙烧—水浸条件驱动 Li、Rb 和 Cs 浸出率预测；面向候选工艺筛选，并保留实验验证边界。</p></div>',unsafe_allow_html=True)
    _,c,_=st.columns([1.05,1,1.05])
    with c,st.container(border=True):
        st.subheader("安全登录"); u=st.text_input("账号"); p=st.text_input("密码",type="password")
        if st.button("进入预测平台",use_container_width=True):
            if hmac.compare_digest(u,expected_user) and hmac.compare_digest(p,expected_password): st.session_state.authenticated=True; st.rerun()
            else: st.error("账号或密码不正确。")
        st.caption("公共演示账号：admin；密码：222333")

if not st.session_state.get("authenticated",False): login(); st.stop()

schema,run,models=load_assets(); features=schema["feature_names"]; ranges=schema["ranges"]; additives=schema["additive_features"]; comps=schema["composition_features"]

with st.sidebar:
    st.markdown("## 🍂 CoreSynergy"); st.caption("Lepidolite digital laboratory"); st.divider()
    for metal in ("Li","Rb","Cs"):
        info=run["best_models"][metal]; h=info["holdout"]; e=info["external"]
        st.markdown(f"**{metal} · {info['best_model']}**"); st.caption(f"文献留出集 R² {h['test_R2']:.3f} · RMSE {h['test_RMSE']:.2f}%"); st.caption(f"外部实验 R² {e['R2']:.3f} · RMSE {e['RMSE']:.2f}%")
    st.divider(); st.caption("31 项输入全部参与模型：6 项矿样成分、19 项添加剂分量和 6 项工艺/总量变量。")
    if st.button("退出登录",use_container_width=True): st.session_state.authenticated=False; st.rerun()

st.markdown('<div class="hero"><div class="eyebrow">CORE SYNERGY · DATA-GUIDED PROCESS SCREENING</div><h1>Li / Rb / Cs 浸出率预测</h1><p>先输入矿样化学组成，再选择一种或多种焙烧添加剂及其相对锂云母的质量比，最后设置焙烧和水浸条件。</p></div>',unsafe_allow_html=True)
tab1,tab2,tab3=st.tabs(["🍂 预测工作台","🧪 全部变量","📖 模型边界"])

with tab1:
    heading(1,"输入矿样化学组成","数据集中实际包含 6 项成分描述符；原表头 Ru(%) 已按 Rb 含量的语义规范。粒度不作为输入，实验与文献比较时应统一控制在 <74 μm。")
    comp_values={}; cols=st.columns(3)
    external_defaults={"Li2O_pct":1.98,"Rb_pct":.211,"Cs_pct":.153,"SiO2_pct":46.701,"Al2O3_pct":36.036,"Fe2O3_pct":3.556}
    for i,name in enumerate(comps):
        r=ranges[name]; value=float(external_defaults.get(name,r["median"])); upper=max(float(r["max"]),value)
        with cols[i%3],st.container(border=True): comp_values[name]=st.number_input(DISPLAY[name],min_value=0.0,max_value=upper*1.5,value=value,step=.01,help=f"文献数据范围：{r['min']:.3g}–{r['max']:.3g}")

    heading(2,"选择全部焙烧添加剂并填写用量","19 种添加剂均来自训练数据。用量定义为“该添加剂/锂云母质量比”；可单选，也可多选构成复合体系。")
    if st.button("清空全部添加剂"):
        for a in additives: st.session_state[f"use_{a}"]=False; st.session_state[f"dose_{a}"]=0.0
        st.rerun()
    add_values={}; cols=st.columns(4)
    for i,name in enumerate(additives):
        r=ranges[name]; st.session_state.setdefault(f"use_{name}",False); st.session_state.setdefault(f"dose_{name}",0.0)
        with cols[i%4],st.container(border=True):
            st.markdown(f'<div class="addname">{i+1:02d} · {DISPLAY[name]}</div><div class="addgroup">{group_of(name)} · 文献上限 {r["max"]:g}</div>',unsafe_allow_html=True)
            use=st.checkbox("使用该添加剂",key=f"use_{name}"); dose=st.number_input(f"{DISPLAY[name]} / 锂云母",min_value=0.0,max_value=float(r["max"]),step=max(.01,round(float(r["max"])/100,3)),key=f"dose_{name}",disabled=not use)
            add_values[name]=float(dose) if use else 0.0
    active=[a for a,v in add_values.items() if v>0]; add_sum=float(sum(add_values.values()))
    st.markdown(f'<div class="formula"><b>当前添加剂体系：</b>{" + ".join(DISPLAY[a] for a in active) if active else "无添加剂"}　　<b>分量之和：</b>{add_sum:.3f}</div>',unsafe_allow_html=True)
    auto=st.toggle("总添加剂/矿石质量比采用分量之和",value=True)
    if auto: total_ratio=add_sum; st.metric(DISPLAY["total_additive_ratio"],f"{total_ratio:.3f}")
    else: total_ratio=st.number_input(DISPLAY["total_additive_ratio"],min_value=float(ranges["total_additive_ratio"]["min"]),max_value=float(ranges["total_additive_ratio"]["max"]),value=float(np.clip(add_sum or 1.0,ranges["total_additive_ratio"]["min"],ranges["total_additive_ratio"]["max"])),step=.05)

    heading(3,"设置焙烧与水浸条件","液固比统一换算为液体相对固体的数值倍率，例如 10:1 输入 10。")
    pnames=["roasting_temp_C","roasting_time_h","liquid_solid_ratio","leaching_temp_C","leaching_time_h"]; defaults=[750.,1.5,10.,50.,1.]; pvals={}; cols=st.columns(5)
    for c,name,d in zip(cols,pnames,defaults):
        r=ranges[name]
        with c,st.container(border=True): pvals[name]=st.number_input(DISPLAY[name],min_value=float(r["min"]),max_value=float(r["max"]),value=float(np.clip(d,r["min"],r["max"])),step=1.0 if "temp" in name else .1,help=f"文献范围：{r['min']:g}–{r['max']:g}")

    row={f:0.0 for f in features}; row.update(comp_values); row.update(add_values); row.update(pvals); row["total_additive_ratio"]=float(total_ratio); X=pd.DataFrame([[row[f] for f in features]],columns=features)
    out_of_range=[DISPLAY.get(f,f) for f in features if float(row[f])<float(ranges[f]["min"]) or float(row[f])>float(ranges[f]["max"])]
    with st.expander("核对全部 31 项输入"):
        st.dataframe(pd.DataFrame({"Feature":[DISPLAY.get(f,f) for f in features],"Value":[row[f] for f in features],"Literature min":[ranges[f]["min"] for f in features],"Literature max":[ranges[f]["max"] for f in features]}),hide_index=True,use_container_width=True,height=600)

    heading(4,"预测三种金属浸出率")
    if st.button("开始预测 Li、Rb、Cs",type="primary",use_container_width=True):
        if out_of_range: st.warning("下列输入超出文献数据范围，属于外推："+"、".join(out_of_range))
        if not active: st.warning("当前未选择添加剂，请确认这是无添加剂对照。")
        result_cols=st.columns(3); palette={"Li":"#178b82","Rb":"#d59632","Cs":"#b65c39"}; export=X.copy()
        for c,metal in zip(result_cols,("Li","Rb","Cs")):
            pred=float(np.clip(models[metal].predict(X)[0],0,100)); rmse=float(run["best_models"][metal]["holdout"]["test_RMSE"]); export[f"{metal}_predicted_recovery_pct"]=pred
            with c:
                st.markdown(f'<div class="result"><div class="rlabel">{metal} predicted recovery</div><div class="rvalue" style="color:{palette[metal]}">{pred:.1f}%</div><div class="rmeta">Holdout RMSE reference: ±{rmse:.2f} percentage points</div></div>',unsafe_allow_html=True); st.progress(pred/100)
        st.error("独立实验显示存在显著矿样域偏移，尤其 Li 与 Cs。结果仅用于候选条件排序，不能替代同矿样的平行实验验证。")
        st.download_button("下载本次输入与预测结果（CSV）",export.to_csv(index=False).encode("utf-8-sig"),"lepidolite_prediction.csv","text/csv")

with tab2:
    st.subheader("模型使用的全部 31 项影响因素")
    rows=[]
    for i,f in enumerate(features,1):
        category="矿样成分" if f in comps else ("焙烧添加剂" if f in additives else "工艺/总量")
        rows.append({"No.":i,"Category":category,"Feature":DISPLAY.get(f,f),"Literature minimum":ranges[f]["min"],"Median":ranges[f]["median"],"Literature maximum":ranges[f]["max"]})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True,height=720)

with tab3:
    st.subheader("模型与适用边界")
    st.markdown("""
- 文献数据共 896 条；Li、Rb、Cs 有效标签分别为 896、509、494 条。
- 按固定随机种子 492 进行 80:20 划分；训练集内部采用 5 折交叉验证和 Optuna-TPE 调参。
- 比较 LightGBM、Random Forest、XGBoost、Stacking、Extra Trees、GBDT 和 SVR；三种目标均依据交叉验证 RMSE 选择 XGBoost。
- 独立验证使用 48 条新实验，不参与调参或模型选择。该验证揭示明显域偏移，因此平台是筛选工具，不是实验替代品。
- 粒度没有进入模型，因为汇总文献通常将其作为固定预处理条件而非系统变量；使用平台时建议将样品统一控制在 <74 μm。
""")
    summary=[]
    for m in ("Li","Rb","Cs"):
        x=run["best_models"][m]; summary.append({"Target":m,"Model":x["best_model"],"Holdout R²":x["holdout"]["test_R2"],"Holdout RMSE":x["holdout"]["test_RMSE"],"External R²":x["external"]["R2"],"External RMSE":x["external"]["RMSE"]})
    st.dataframe(pd.DataFrame(summary).style.format({"Holdout R²":"{:.3f}","Holdout RMSE":"{:.2f}","External R²":"{:.3f}","External RMSE":"{:.2f}"}),hide_index=True,use_container_width=True)
