from __future__ import annotations

import hashlib
import hmac
import html
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from predictor import (
    ADDITIVES, DISPLAY, METALS, ORE, PROCESS, RAW_FEATURES,
    audit_frame, evidence_table, load_bundle, predict_frame,
)

BASE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_BUNDLE_DIR", str(BASE / "model_bundle")))
COLORS = {"Li": "#167d78", "Rb": "#b67916", "Cs": "#a95377"}
st.set_page_config(page_title="CoreSynergy | Li · Rb · Cs", page_icon="🍂", layout="wide")
st.markdown("""<style>
.stApp {color:#243b3a;background:linear-gradient(125deg,#fffdf8 0%,#faf1e1 55%,#eef7f2 100%)}
[data-testid="stSidebar"] {background:#f4ebdd;border-right:1px solid #d8c9b0}
[data-testid="stHeader"] {background:rgba(255,253,248,.9)}
.hero {padding:1.5rem 1.7rem;margin-bottom:1rem;border:1px solid #ddc9ab;border-radius:20px;background:linear-gradient(110deg,#fff,#fcf0d8 65%,#e4f3ee)}
.eyebrow {color:#9b5539;letter-spacing:.13em;font-size:.76rem;font-weight:750}
.hero h1 {font-size:2rem;color:#244d48;margin:.3rem 0}.hero p {max-width:960px;color:#596e65;line-height:1.6}
.score {padding:.75rem .85rem;margin:.6rem 0;border:1px solid #d8c9b0;border-radius:12px;background:#fffaf1}
.score b {color:#274b47}.score small {color:#596e65;line-height:1.7}
.result {padding:1.1rem 1.3rem;background:white;border:1px solid #cdbda5;border-radius:16px}
.result .value {font-size:2.5rem;font-weight:750}.result .label {font-size:.94rem;color:#596e65}
.result .algorithm {font-size:.85rem;color:#596e65;line-height:1.6}
.stButton>button[kind="primary"] {background:#167d78;color:white;border:1px solid #115f5b}
</style>""", unsafe_allow_html=True)


def hero():
    st.markdown('<div class="hero"><div class="eyebrow">CORE SYNERGY · LEPIDOLITE PROCESS SCREENING</div>'
                '<h1>Li / Rb / Cs recovery prediction</h1>'
                '<p>Enter measured ore composition, a single or combined roasting-additive system, '
                'and roasting–water-leaching conditions. Compare predictions across ore samples with '
                'the same reproducible model.</p></div>', unsafe_allow_html=True)


def login():
    try:
        secret = st.secrets.get("auth", {})
    except FileNotFoundError:
        secret = {}
    username = str(secret.get("username") or os.getenv("APP_USERNAME") or "")
    password = str(secret.get("password") or os.getenv("APP_PASSWORD") or "")
    if not username or not password:
        hero()
        st.error("Administrator setup required: configure APP_USERNAME and APP_PASSWORD, or .streamlit/secrets.toml.")
        st.stop()
    hero()
    _, middle, _ = st.columns([1, 1.3, 1])
    with middle, st.form("sign_in"):
        st.subheader("Sign in")
        entered_user = st.text_input("Account")
        entered_password = st.text_input("Password", type="password")
        if st.form_submit_button("Open prediction workspace", use_container_width=True):
            user_ok = hmac.compare_digest(entered_user.encode(), username.encode())
            password_ok = hmac.compare_digest(entered_password.encode(), password.encode())
            if user_ok and password_ok:
                st.session_state.authenticated = True
                st.rerun()
            st.error("Incorrect account or password.")
    st.stop()


@st.cache_resource
def assets(directory: str, manifest_hash: str):
    return load_bundle(Path(directory))


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def feature_input(name: str, default: float, ranges: dict):
    r = ranges[name]
    kwargs = {"min_value": 0.0, "value": float(default), "key": f"input_{name}",
              "step": 1.0 if "temp" in name else 0.01,
              "help": f"Observed modeling-data range: {r['min']:.4g}–{r['max']:.4g}. Inputs outside this range are flagged."}
    if name in ORE:
        kwargs["max_value"] = 100.0
        kwargs["format"] = "%.3f"
    return st.number_input(DISPLAY[name], **kwargs)


if not st.session_state.get("authenticated", False):
    login()

try:
    manifest_path = MODEL_DIR / "manifest.json"
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest, models, uncertainty_models = assets(str(MODEL_DIR), manifest_hash)
    evidence = evidence_table(manifest)
except (FileNotFoundError, ValueError, KeyError, ImportError) as error:
    hero()
    st.error(f"Model package is unavailable or incompatible: {error}")
    st.caption("Install the complete model_bundle directory and matching model_features.py from this analysis run.")
    st.stop()

ranges = manifest["ranges"]
with st.sidebar:
    st.markdown("## 🍂 CoreSynergy")
    st.caption("Lepidolite digital laboratory")
    st.divider()
    st.subheader("Model evidence")
    for metal in METALS:
        item = manifest["best_models"][metal]
        st.markdown(f'<div class="score"><b>{metal}</b><br>'
                    f'<small>Training 5-fold CV R²: {item["cv"]["r2_mean"]:.3f}<br>'
                    f'Test-set R²: {item["test"]["r2"]:.3f}</small></div>', unsafe_allow_html=True)
    st.caption("CV R² is the arithmetic mean across five held-out folds. Full metrics and model provenance are available in Model evidence.")
    st.divider()
    st.caption("31 raw inputs · 6 ore contents · 19 additive components · 1 total ratio · 5 process variables")
    if st.button("Sign out", use_container_width=True):
        st.session_state.clear()
        st.rerun()

hero()
prediction_tab, batch_tab, evidence_tab, method_tab = st.tabs([
    "Prediction workspace", "Compare ore samples", "Model evidence", "Method notes"])

with prediction_tab:
    st.subheader("1 · Ore composition")
    st.caption("Enter the six measured composition descriptors in mass percent. Ore grade is not encoded as a category.")
    defaults = {"Li2O": 1.98, "Rb": .211, "Cs": .153, "SiO2": 46.701, "Al2O3": 36.036, "Fe2O3": 3.556}
    row = {name: 0.0 for name in RAW_FEATURES}
    for start in (0, 3):
        for column, name in zip(st.columns(3), ORE[start:start+3]):
            with column:
                row[name] = feature_input(name, defaults[name], ranges)

    st.subheader("2 · Roasting additives")
    st.caption("Select one reagent for a single-additive system, or multiple reagents for a combined system. All doses are additive / ore mass ratios.")
    selected = st.multiselect("Select from all 19 additives", ADDITIVES,
                             default=["NaCl", "CaCl2"], format_func=lambda key: DISPLAY[key], key="selected_additives")
    for start in range(0, len(selected), 3):
        for column, name in zip(st.columns(3), selected[start:start+3]):
            with column:
                default = {"NaCl": .2, "CaCl2": .3}.get(name, .1)
                row[name] = feature_input(name, default, ranges)
    active = [name for name in ADDITIVES if row[name] > 0]
    component_sum = sum(row[name] for name in ADDITIVES)
    system = "No additive" if not active else "Single additive" if len(active) == 1 else "Combined additives"
    st.info(f"{system}: " + (" + ".join(DISPLAY[name] for name in active) or "control") + f" · Component sum = {component_sum:.3f}")
    auto_total = st.toggle("Calculate total additive / ore ratio from the component sum", value=True)
    if auto_total:
        row["Total_ratio"] = component_sum
        st.metric(DISPLAY["Total_ratio"], f"{component_sum:.3f}")
    else:
        row["Total_ratio"] = feature_input("Total_ratio", component_sum, ranges)
        if not np.isclose(row["Total_ratio"], component_sum, atol=1e-6):
            st.warning("The entered total differs from the component sum. Both values will be retained in the exported record.")

    st.subheader("3 · Roasting and water-leaching conditions")
    st.caption("Liquid:solid is entered directly: 0.8 means 0.8:1; 10 means 10:1. Time units are hours.")
    process_defaults = {"Roast_temp": 750.0, "Roast_time": .75, "Liquid_solid": 3.0, "Leach_temp": 25.0, "Leach_time": 1.0}
    for column, name in zip(st.columns(5), PROCESS):
        with column:
            row[name] = feature_input(name, process_defaults[name], ranges)
    raw = pd.DataFrame([row], columns=RAW_FEATURES)
    audit = audit_frame(raw, ranges).iloc[0]
    if audit["outside_observed_range"]:
        names = audit["outside_observed_range"].split("; ")
        st.warning("Outside the observed modeling-data range: " + ", ".join(DISPLAY[name] for name in names) +
                   ". This prediction involves extrapolation; confirm it on the same ore sample.")
    fingerprint = hashlib.sha256(raw.to_json().encode()).hexdigest()
    st.subheader("4 · Predict Li, Rb and Cs recoveries")
    if st.button("Run prediction", type="primary", use_container_width=True):
        try:
            st.session_state.prediction_result = predict_frame(raw, models, manifest, uncertainty_models)
            st.session_state.prediction_input_hash = fingerprint
        except ValueError as error:
            st.error(str(error))
    saved = st.session_state.get("prediction_result")
    if saved is not None and st.session_state.get("prediction_input_hash") == fingerprint:
        for column, metal in zip(st.columns(3), METALS):
            prediction = saved.iloc[0][f"{metal}_predicted_display_pct"]
            raw_prediction = saved.iloc[0][f"{metal}_predicted_raw_pct"]
            lower = saved.iloc[0][f"{metal}_pi_lower_pct"]
            upper = saved.iloc[0][f"{metal}_pi_upper_pct"]
            sd = saved.iloc[0][f"{metal}_ensemble_sd_pp"]
            confidence = saved.iloc[0][f"{metal}_confidence_flag"]
            algorithm = manifest["best_models"][metal]["algorithm"]
            with column:
                st.markdown(f'<div class="result"><div class="label">{metal} predicted recovery</div>'
                            f'<div class="value" style="color:{COLORS[metal]}">{prediction:.2f}%</div>'
                            f'<div class="algorithm">Selected algorithm: {html.escape(algorithm)}<br>'
                            f'95% PI: {lower:.2f}–{upper:.2f}%<br>'
                            f'Ensemble SD: {sd:.2f} pp<br>'
                            f'<span style="color:{"#a33b32" if str(confidence).startswith("Low confidence") else "#39735c"}">{html.escape(str(confidence))}</span><br>'
                            f'Raw prediction: {raw_prediction:.2f}%</div></div>', unsafe_allow_html=True)
                st.progress(float(prediction) / 100)
        if saved["confidence_flag"].eq("Low confidence").any():
            st.warning("Low-confidence warning: at least one target has high ensemble variance, a wide prediction interval, or an input outside the observed modeling-data domain. Confirm the condition with a repeat experiment.")
        st.caption("Prediction intervals use five-fold ensemble variance combined with out-of-fold RMSE. Cards are limited to 0–100% for display; raw predictions and interval diagnostics are preserved in the CSV.")
        st.download_button("Download inputs and predictions (CSV)", csv_bytes(saved), "lepidolite_prediction.csv", "text/csv")
    elif saved is not None:
        st.info("Inputs have changed. Run prediction to update the results.")

with batch_tab:
    st.subheader("Compare different ore samples")
    st.write("Use one row per ore sample and process condition. The template contains the current 31 inputs. You can add Ore_ID and Formulation_ID columns to retain sample identity.")
    template = raw.copy()
    template.insert(0, "Ore_ID", "Ore A")
    template.insert(1, "Formulation_ID", "F1")
    st.download_button("Download current-input CSV template", csv_bytes(template), "ore_comparison_template.csv", "text/csv")
    uploaded = st.file_uploader("Upload a CSV with the 31 named input columns", type=["csv"])
    if uploaded is not None:
        payload = uploaded.getvalue()
        if len(payload) > 5_000_000:
            st.error("Please upload a CSV smaller than 5 MB.")
        else:
            try:
                batch = pd.read_csv(io.BytesIO(payload))
                if len(batch) > 10000:
                    raise ValueError("Please limit one comparison to 10,000 rows.")
                predictions = predict_frame(batch, models, manifest, uncertainty_models)
                keep = [key for key in ["Ore_ID", "Formulation_ID"] if key in predictions]
                keep += [f"{metal}_predicted_raw_pct" for metal in METALS]
                keep += [f"{metal}_pi_lower_pct" for metal in METALS]
                keep += [f"{metal}_pi_upper_pct" for metal in METALS]
                keep += [f"{metal}_ensemble_sd_pp" for metal in METALS]
                keep += [f"{metal}_confidence_flag" for metal in METALS]
                keep += ["confidence_flag", "additive_system", "outside_observed_range", "total_minus_components"]
                st.dataframe(predictions[keep], hide_index=True, use_container_width=True)
                outside = predictions["outside_observed_range"].ne("").sum()
                low_conf = predictions["confidence_flag"].eq("Low confidence").sum()
                if outside or low_conf:
                    st.warning(f"{low_conf} sample(s) are low-confidence and {outside} sample(s) contain inputs outside the observed range. See the interval and flag columns before interpreting differences between ore samples.")
                st.download_button("Download all inputs, audit flags and predictions", csv_bytes(predictions), "ore_comparison_predictions.csv", "text/csv")
                st.caption("Uploaded rows are used only for prediction. They do not alter model fitting, tuning or selection.")
            except (ValueError, pd.errors.ParserError, UnicodeError) as error:
                st.error(str(error))

with evidence_tab:
    st.subheader("Selected-model performance")
    st.dataframe(evidence, hide_index=True, use_container_width=True)
    st.download_button("Download full performance table", csv_bytes(evidence), "selected_model_performance.csv", "text/csv")
    st.caption("RMSE is in percentage points (pp). CV statistics use held-out training folds. Prediction intervals use five-fold ensemble variance plus out-of-fold RMSE; all reported performance uses raw predictions before display clipping.")
    st.info("Algorithms and hyperparameters were selected using five-fold cross-validation within the training partition. Prediction intervals are generated from five deployment-fold models and calibrated with out-of-fold RMSE. The 20% test partition and separate experimental workbook were excluded from tuning and model selection.")
    with st.expander("Model provenance and input ranges"):
        provenance = {key: value for key, value in manifest.items() if key not in {"ranges", "best_models", "raw_feature_names"}}
        st.json(provenance)
        st.dataframe(pd.DataFrame(ranges).T.rename_axis("Feature"), use_container_width=True)
        st.download_button("Download model manifest", json.dumps(manifest, indent=2, ensure_ascii=False).encode(), "manifest.json", "application/json")

with method_tab:
    st.subheader("Method and scope")
    modeling_n = manifest["modeling_rows"]
    external_n = manifest.get("external_rows", 8)
    st.markdown(f"""
- All **{modeling_n} original dataset records** form the modeling pool. Each target uses every available measured label; missing recovery labels are not invented or imputed.
- The complete modeling workbook is split **80:20** with random_state 492 for each metal. Five-fold shuffled cross-validation runs only within the training partition.
- Seven algorithms are compared: LightGBM, Random Forest, XGBoost, Stacking, Extra Trees, GBDT and SVR. Hyperparameters are optimized with Optuna-TPE using only five-fold cross-validation inside each training partition; selection uses the lowest mean CV RMSE.
- The separate experimental workbook (**{external_n} experimental records**) is reserved for external validation across ore samples. It is excluded from fitting, tuning and selection.
- Six ore-composition contents, 19 additive component ratios, the total additive ratio and five process conditions are retained. Feature transformation uses the same module as model training.
- Total additive ratio is calculated from the component doses by default. An explicit override preserves any discrepancy for traceability.
    - Model evaluation uses raw predictions. Display cards alone are limited to 0–100%. The GUI reports a 95% prediction interval from five-fold ensemble variance plus out-of-fold RMSE, the ensemble SD, interval width, and a low-confidence flag.
- Random record-level generalization and cross-ore generalization are different tests. Grouping identical inputs prevents duplicate leakage, but does not replace leave-study-out or prospective validation.
""")
    fits = [f"{metal}: {manifest['best_models'][metal].get('deployment_fit_n', 'see manifest')} records"
            for metal in METALS]
    st.write("Deployment fitting set — " + "; ".join(fits) + ".")
    st.caption("For candidate-condition screening. A low-confidence warning is raised for domain extrapolation, high ensemble variance, or an unusually wide interval. Experimental confirmation on the intended ore sample remains the basis for process decisions.")

