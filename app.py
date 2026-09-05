from __future__ import annotations
import hmac, json, os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
METALS = ("Li", "Rb", "Cs")
st.set_page_config(page_title="CoreSynergy | Lepidolite predictor", page_icon="🍂", layout="wide")

st.markdown("""<style>
:root{--ink:#243b3a;--muted:#687873;--teal:#167d78;--deep:#165c59;--amber:#d9952f;--copper:#b85e3f}
.stApp{color:var(--ink);background:radial-gradient(circle at 8% 4%,rgba(217,149,47,.18),transparent 25%),radial-gradient(circle at 93% 8%,rgba(22,125,120,.15),transparent 27%),linear-gradient(135deg,#fffdf8 0%,#f7efe0 53%,#edf7f3 100%)}
.stApp:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.07;background-image:linear-gradient(rgba(22,92,89,.18) 1px,transparent 1px),linear-gradient(90deg,rgba(22,92,89,.18) 1px,transparent 1px);background-size:52px 52px;mask-image:linear-gradient(to bottom,black,transparent 72%)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#fff9ec,#f2e7d2 57%,#e8f4ef);border-right:1px solid rgba(184,94,63,.18)}
[data-testid="stHeader"]{background:rgba(255,253,248,.75)}
.hero{position:relative;overflow:hidden;padding:1.45rem 1.7rem;border-radius:24px;background:linear-gradient(118deg,rgba(255,255,255,.96),rgba(255,245,220,.94) 56%,rgba(226,246,239,.94));border:1px solid rgba(184,94,63,.18);box-shadow:0 16px 46px rgba(78,65,41,.10);margin-bottom:1rem}
.hero:after{content:"";position:absolute;right:-58px;top:-92px;width:270px;height:270px;border:28px solid rgba(217,149,47,.14);border-radius:50%;box-shadow:0 0 0 24px rgba(22,125,120,.07),0 0 0 52px rgba(184,94,63,.05)}
.eyebrow{color:var(--copper);font-weight:800;letter-spacing:.13em;font-size:.75rem}.hero h1{margin:.28rem 0 0;color:#244d48;font-size:2.08rem}.hero p{margin:.55rem 0 0;color:#64736f;max-width:900px;line-height:1.6}
.step{display:flex;align-items:center;gap:.65rem;margin:1.15rem 0 .55rem}.badge{display:inline-flex;width:32px;height:32px;align-items:center;justify-content:center;border-radius:10px;color:white;font-weight:800;background:linear-gradient(135deg,var(--copper),var(--amber));box-shadow:0 6px 16px rgba(184,94,63,.20)}.step b{font-size:1.16rem;color:#304f4c}.note{color:#6b7975;margin:-.18rem 0 .65rem;font-size:.90rem}
div[data-testid="stVerticalBlockBorderWrapper"]{background:rgba(255,255,255,.78);border-color:rgba(184,94,63,.16)!important;box-shadow:0 8px 24px rgba(80,67,45,.05);border-radius:16px}
div[data-testid="stNumberInput"] input,div[data-testid="stTextInput"] input{background:rgba(255,255,255,.96)!important;border-color:rgba(22,125,120,.22)!important}
.formula{padding:.8rem 1rem;border-radius:14px;background:linear-gradient(90deg,rgba(217,149,47,.13),rgba(22,125,120,.09));border:1px solid rgba(22,125,120,.16);color:#3d5753}.addname{font-weight:800;color:#31514e}.addgroup{font-size:.76rem;color:#96683f;margin-bottom:.3rem}
.score{padding:.8rem .85rem;border-radius:14px;background:rgba(255,255,255,.72);border:1px solid rgba(22,92,89,.12);margin:.45rem 0}.score-title{font-weight:800;color:#274b47}.score-line{font-size:.80rem;color:#667671;margin-top:.18rem}.result{border-radius:19px;padding:1.15rem 1.30rem;background:rgba(255,255,255,.94);border:1px solid rgba(184,94,63,.17);box-shadow:0 12px 30px rgba(80,64,39,.08)}.rlabel{font-size:.88rem;color:#697a76}.rvalue{font-size:2.45rem;font-weight:800;line-height:1.08;margin:.28rem 0}.rmeta{font-size:.80rem;color:#6e7e7a;line-height:1.5}
.stButton>button,.stFormSubmitButton>button{border:0;border-radius:12px;font-weight:800;color:white;background:linear-gradient(90deg,#b85e3f,#d9952f 52%,#167d78);box-shadow:0 9px 22px rgba(184,94,63,.17)}
</style>""", unsafe_allow_html=True)

DISPLAY = {"Li2O_pct":"Li₂O content (%)","Rb_pct":"Rb content (%)","Cs_pct":"Cs content (%)","SiO2_pct":"SiO₂ content (%)","Al2O3_pct":"Al₂O₃ content (%)","Fe2O3_pct":"Fe₂O₃ content (%)","H2SO4":"H₂SO₄","HCl":"HCl","K2S2O7":"K₂S₂O₇","KHSO4":"KHSO₄","FeSO4_7H2O":"FeSO₄·7H₂O","KOH":"KOH","CaO":"CaO","NaCl":"NaCl","CaCl2":"CaCl₂","SLS":"SLS (sodium lignosulfonate)","NaOH":"NaOH","CaOH2":"Ca(OH)₂","NH4_2SO4":"(NH₄)₂SO₄","Na2SO4":"Na₂SO₄","CaSO4":"CaSO₄","CaCO3":"CaCO₃","K2SO4":"K₂SO₄","NaHSO4":"NaHSO₄","C":"C","total_additive_ratio":"Total additive / ore mass ratio","roasting_temp_C":"Roasting temperature (°C)","roasting_time_h":"Roasting time (h)","liquid_solid_ratio":"Liquid-to-solid ratio","leaching_temp_C":"Water-leaching temperature (°C)","leaching_time_h":"Water-leaching time (h)"}
GROUPS={"Acid / acid salt":{"H2SO4","HCl","KHSO4","NaHSO4"},"Sulfate / pyrosulfate":{"K2S2O7","FeSO4_7H2O","NH4_2SO4","Na2SO4","CaSO4","K2SO4"},"Chloride":{"NaCl","CaCl2"},"Alkali / calcium":{"KOH","CaO","NaOH","CaOH2","CaCO3"},"Auxiliary":{"SLS","C"}}

def feature_engineering(frame, raw_features):
    x=frame[raw_features].copy(); a=["H2SO4","HCl","K2S2O7","KHSO4","FeSO4_7H2O","KOH","CaO","NaCl","CaCl2","SLS","NaOH","CaOH2","NH4_2SO4","Na2SO4","CaSO4","CaCO3","K2SO4","NaHSO4","C"]; eps=1e-6
    v=x[a].fillna(0); total=v.sum(axis=1); active=v.gt(0)
    x["additive_sum"]=total; x["active_additive_count"]=active.sum(axis=1); x["is_single_additive"]=active.sum(axis=1).eq(1).astype(float); x["is_compound_additive"]=active.sum(axis=1).gt(1).astype(float); x["recorded_minus_component_sum"]=x["total_additive_ratio"]-total
    for n in a:x[f"frac_{n}"]=v[n]/(total+eps)
    x["sulfate_family"]=v[["H2SO4","K2S2O7","KHSO4","FeSO4_7H2O","NH4_2SO4","Na2SO4","CaSO4","K2SO4","NaHSO4"]].sum(axis=1); x["chloride_family"]=v[["HCl","NaCl","CaCl2"]].sum(axis=1); x["alkali_family"]=v[["KOH","NaOH","CaOH2"]].sum(axis=1); x["calcium_family"]=v[["CaO","CaCl2","CaOH2","CaSO4","CaCO3"]].sum(axis=1); x["acid_family"]=v[["H2SO4","HCl","KHSO4","NaHSO4"]].sum(axis=1)
    x["fe_na_sulfate_synergy"]=v["FeSO4_7H2O"]*v["Na2SO4"]; x["roasting_severity"]=x["roasting_temp_C"]*np.log1p(x["roasting_time_h"].clip(lower=0)); x["leaching_severity"]=x["leaching_temp_C"]*np.log1p(x["leaching_time_h"].clip(lower=0)); x["additive_thermal_load"]=x["total_additive_ratio"]*x["roasting_temp_C"]; x["liquid_time_exposure"]=x["liquid_solid_ratio"]*x["leaching_time_h"]
    x["Li2O_over_SiO2"]=x["Li2O_pct"]/(x["SiO2_pct"]+eps); x["Li2O_over_Al2O3"]=x["Li2O_pct"]/(x["Al2O3_pct"]+eps); x["Rb_over_Li2O"]=x["Rb_pct"]/(x["Li2O_pct"]+eps); x["Cs_over_Li2O"]=x["Cs_pct"]/(x["Li2O_pct"]+eps); x["SiO2_over_Al2O3"]=x["SiO2_pct"]/(x["Al2O3_pct"]+eps)
    return x.replace([np.inf,-np.inf],np.nan)

@st.cache_resource
def load_assets():
    schema=json.loads((MODEL_DIR/"feature_schema.json").read_text(encoding="utf-8")); run=json.loads((MODEL_DIR/"run_summary.json").read_text(encoding="utf-8")); models={m:joblib.load(MODEL_DIR/f"best_{m}.joblib") for m in METALS}; return schema,run,models

def login():
    try: secret=st.secrets.get("auth",{})
    except FileNotFoundError: secret={}
    expected_u=str(secret.get("username") or os.getenv("APP_USERNAME") or "admin"); expected_p=str(secret.get("password") or os.getenv("APP_PASSWORD") or "222333")
    st.markdown('<div class="hero"><div class="eyebrow">AUTUMN INTELLIGENCE · GREEN METALLURGY</div><h1>🍂 Lepidolite Li/Rb/Cs leaching predictor</h1><p>Screen ore composition, all 19 roasting-additive inputs and process conditions in one reproducible workflow.</p></div>',unsafe_allow_html=True)
    _,mid,_=st.columns([1.1,1,1.1])
    with mid,st.container(border=True):
        st.subheader("Secure sign-in"); u=st.text_input("Account"); p=st.text_input("Password",type="password")
        if st.button("Enter prediction workspace",use_container_width=True):
            if hmac.compare_digest(u,expected_u) and hmac.compare_digest(p,expected_p): st.session_state.authenticated=True; st.rerun()
            else: st.error("Incorrect account or password.")
        st.caption("Demo account: admin · password: 222333")

if not st.session_state.get("authenticated",False): login(); st.stop()
schema,run,models=load_assets(); features=schema["raw_feature_names"]; ranges=schema["ranges"]; additives=schema["additive_features"]; composition=schema["composition_features"]
with st.sidebar:
    st.markdown("## 🍂 CoreSynergy"); st.caption("Lepidolite digital laboratory"); st.divider(); st.markdown("### Model evidence")
    for m in METALS:
        i=run["best_models"][m]; st.markdown(f'<div class="score"><div class="score-title">{m} · {i["best_model"]}</div><div class="score-line">Training 5-fold CV R² · {i["cv"]["cv_R2_oof"]:.3f}</div><div class="score-line">Test-set R² · {i["test"]["test_R2"]:.3f}</div></div>',unsafe_allow_html=True)
    st.divider(); st.caption("All 31 raw inputs are retained. Derived additive-family, fraction and process-intensity features are constructed internally.")
    if st.button("Sign out",use_container_width=True): st.session_state.authenticated=False; st.rerun()

st.markdown('<div class="hero"><div class="eyebrow">CORE SYNERGY · DATA-GUIDED PROCESS SCREENING</div><h1>Li / Rb / Cs recovery prediction</h1><p>Enter the six measured ore-composition contents, choose any single or combined additive system, and set roasting–leaching conditions. The fitted models return recoveries in the physical 0–100% interval.</p></div>',unsafe_allow_html=True)
prediction_tab,evidence_tab,method_tab=st.tabs(["🍂 Prediction workspace","📊 Model evidence","📘 Method notes"])
with prediction_tab:
    st.markdown('<div class="step"><span class="badge">1</span><b>Ore composition</b></div><div class="note">Use the six measured composition descriptors. Particle size is controlled as a pretreatment condition and is not a model input.</div>',unsafe_allow_html=True)
    defaults={"Li2O_pct":1.98,"Rb_pct":.211,"Cs_pct":.153,"SiO2_pct":46.701,"Al2O3_pct":36.036,"Fe2O3_pct":3.556}; comp={}
    for start in range(0,len(composition),3):
        cols=st.columns(3)
        for col,n in zip(cols,composition[start:start+3]):
            r=ranges[n]; d=float(np.clip(defaults.get(n,r["median"]),r["min"],max(r["min"],r["max"])))
            with col,st.container(border=True): comp[n]=st.number_input(DISPLAY[n],min_value=0.,max_value=float(max(r["max"]*1.5,d*1.25)),value=d,step=.01,help=f"Observed range: {r['min']:.3g}–{r['max']:.3g}")
    st.markdown('<div class="step"><span class="badge">2</span><b>Roasting additives</b></div><div class="note">All 19 additive columns are available. Tick a reagent to activate it, then enter its additive-to-ore mass ratio.</div>',unsafe_allow_html=True)
    if st.button("Clear additive selections"):
        for n in additives: st.session_state[f"use_{n}"]=False; st.session_state[f"dose_{n}"]=0.
        st.rerun()
    vals={}
    for start in range(0,len(additives),4):
        cols=st.columns(4)
        for idx,(col,n) in enumerate(zip(cols,additives[start:start+4]),start=start+1):
            r=ranges[n]; st.session_state.setdefault(f"use_{n}",False); st.session_state.setdefault(f"dose_{n}",0.)
            with col,st.container(border=True):
                st.markdown(f'<div class="addname">{idx:02d} · {DISPLAY[n]}</div><div class="addgroup">{next(g for g,members in GROUPS.items() if n in members)} · observed max {r["max"]:g}</div>',unsafe_allow_html=True)
                enabled=st.checkbox("Use additive",key=f"use_{n}"); dose=st.number_input("Additive / ore",min_value=0.,max_value=float(r["max"]),step=max(.01,round(float(r["max"])/100,3)),key=f"dose_{n}",disabled=not enabled); vals[n]=float(dose) if enabled else 0.
    active=[n for n,v in vals.items() if v>0]; total=sum(vals.values()); active_text=" + ".join(DISPLAY[n] for n in active) if active else "No additive (control)"
    st.markdown(f'<div class="formula"><b>Active system:</b> {active_text} · <b>Component sum:</b> {total:.3f}</div>',unsafe_allow_html=True)
    auto=st.toggle("Set total additive / ore ratio equal to component sum",value=True)
    total_ratio=float(total) if auto else st.number_input(DISPLAY["total_additive_ratio"],min_value=0.,max_value=max(float(ranges["total_additive_ratio"]["max"]),float(total)),value=float(total),step=.05)
    if auto: st.metric(DISPLAY["total_additive_ratio"],f"{total_ratio:.3f}")
    st.markdown('<div class="step"><span class="badge">3</span><b>Roasting and water-leaching conditions</b></div><div class="note">Liquid-to-solid ratio is entered as liquid:solid (for example, 10 means 10:1).</div>',unsafe_allow_html=True)
    pdfl={"roasting_temp_C":750.,"roasting_time_h":1.5,"liquid_solid_ratio":10.,"leaching_temp_C":50.,"leaching_time_h":1.}; proc={}
    cols=st.columns(5)
    for col,n in zip(cols,schema["process_features"]):
        r=ranges[n]; d=float(np.clip(pdfl[n],r["min"],r["max"]))
        with col,st.container(border=True): proc[n]=st.number_input(DISPLAY[n],min_value=float(r["min"]),max_value=float(r["max"]),value=d,step=1. if "temp" in n else .1,help=f"Observed range: {r['min']:g}–{r['max']:g}")
    row={f:0. for f in features}; row.update(comp); row.update(vals); row.update(proc); row["total_additive_ratio"]=total_ratio
    raw=pd.DataFrame([[row[f] for f in features]],columns=features); model_frame=feature_engineering(raw,features).reindex(columns=schema["model_feature_names"])
    st.markdown('<div class="step"><span class="badge">4</span><b>Predict Li, Rb and Cs recoveries</b></div>',unsafe_allow_html=True)
    if st.button("Run prediction",type="primary",use_container_width=True):
        cols=st.columns(3); export=raw.copy(); colors={"Li":"#167d78","Rb":"#d9952f","Cs":"#b85e3f"}
        for col,m in zip(cols,METALS):
            pred=float(np.clip(models[m].predict(model_frame)[0],0.,100.)); i=run["best_models"][m]; export[f"{m}_predicted_recovery_pct"]=pred
            with col:
                st.markdown(f'<div class="result"><div class="rlabel">{m} predicted recovery</div><div class="rvalue" style="color:{colors[m]}">{pred:.1f}%</div><div class="rmeta">Best model: {i["best_model"]}<br>Training CV R²: {i["cv"]["cv_R2_oof"]:.3f}<br>Test-set R²: {i["test"]["test_R2"]:.3f}</div></div>',unsafe_allow_html=True); st.progress(pred/100)
        st.download_button("Download this input and prediction (CSV)",export.to_csv(index=False).encode("utf-8-sig"),"lepidolite_prediction.csv","text/csv")
with evidence_tab:
    st.subheader("Selected-model evidence"); rows=[]
    for m in METALS:
        i=run["best_models"][m]; rows.append({"Target":m,"Best model":i["best_model"],"Training CV R²":i["cv"]["cv_R2_oof"],"Test-set R²":i["test"]["test_R2"],"Test RMSE (percentage points)":i["test"]["test_RMSE"]})
    st.dataframe(pd.DataFrame(rows).style.format({"Training CV R²":"{:.3f}","Test-set R²":"{:.3f}","Test RMSE (percentage points)":"{:.2f}"}),hide_index=True,use_container_width=True)
    st.info("Seven algorithms were ranked by mean five-fold CV RMSE within the 80% training partition. Test-set labels were not used for model selection; the 20% test partition is shown only as a final descriptive check.")
with method_tab:
    st.subheader("Method and scope")
    st.markdown("""- The complete 944-row workbook is the modeling pool. For each target, available rows are randomly split 80:20 into a training partition and a test partition using a fixed seed.
- LightGBM, random forest, XGBoost, stacking, extremely randomized trees, GBDT and SVR are compared. Hyperparameters are taken from the Optuna-TPE search, and five-fold shuffled cross-validation is performed within the training partition.
- Model selection uses the lowest mean CV RMSE. The selected XGBoost model is refitted using all rows available for that target before screening.
- Six measured ore-composition contents, 19 individual additive-to-ore ratios and five process descriptors are retained as raw inputs. Derived family totals, component fractions and process-severity descriptors are constructed internally.
- Particle size is not entered because the source literature generally treats grinding as a fixed pretreatment or reports non-comparable size descriptors. Keep feed consistently below 74 μm (200 mesh) and confirm the selected condition on the same ore batch.
- Predictions are constrained to 0–100% for display and are intended for process-condition screening, not a substitute for experimental confirmation.""")

