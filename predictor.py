"""Inference and audit helpers. Models accept the same raw inputs as training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from model_features import ADDITIVES, ORE, PROCESS, RAW_FEATURES

METALS = ("Li", "Rb", "Cs")
DISPLAY = {
    "Li2O": "Li₂O content (%)", "Rb": "Rb content (%)", "Cs": "Cs content (%)",
    "SiO2": "SiO₂ content (%)", "Al2O3": "Al₂O₃ content (%)", "Fe2O3": "Fe₂O₃ content (%)",
    "H2SO4": "H₂SO₄", "HCl": "HCl", "K2S2O7": "K₂S₂O₇", "KHSO4": "KHSO₄",
    "FeSO4_7H2O": "FeSO₄·7H₂O", "KOH": "KOH", "CaO": "CaO", "NaCl": "NaCl",
    "CaCl2": "CaCl₂", "SLS": "SLS (sodium lignosulfonate)", "NaOH": "NaOH",
    "CaOH2": "Ca(OH)₂", "NH4_2SO4": "(NH₄)₂SO₄", "Na2SO4": "Na₂SO₄",
    "CaSO4": "CaSO₄", "CaCO3": "CaCO₃", "K2SO4": "K₂SO₄", "NaHSO4": "NaHSO₄", "C": "C",
    "Total_ratio": "Total additive / ore mass ratio", "Roast_temp": "Roasting temperature (°C)",
    "Roast_time": "Roasting time (h)", "Liquid_solid": "Liquid-to-solid ratio (liquid:solid)",
    "Leach_temp": "Water-leaching temperature (°C)", "Leach_time": "Water-leaching time (h)",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_bundle(directory: Path):
    """Load only deployment-local artifacts; never deserialize uploaded files."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["raw_feature_names"] != RAW_FEATURES:
        raise ValueError("Model feature order does not match model_features.py.")
    if not all(name in manifest["ranges"] for name in RAW_FEATURES):
        raise ValueError("The model manifest has incomplete input ranges.")
    models = {}
    for metal in METALS:
        path = directory / f"{metal}.joblib"
        expected = manifest.get("model_sha256", {}).get(metal)
        if expected and sha256_file(path) != expected:
            raise ValueError(f"{metal} model checksum does not match the manifest.")
        models[metal] = joblib.load(path)
    return manifest, models


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [name for name in RAW_FEATURES if name not in frame]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    if not len(frame):
        raise ValueError("The input contains no samples.")
    if frame.columns.duplicated().any():
        raise ValueError("Column names must be unique.")
    values = frame.loc[:, RAW_FEATURES].apply(pd.to_numeric, errors="coerce")
    invalid = ~np.isfinite(values.to_numpy(dtype=float))
    if invalid.any():
        bad = [str(name) for name in values.columns[invalid.any(axis=0)]]
        raise ValueError("Enter a finite numeric value in every input: " + ", ".join(bad))
    if (values < 0).any().any():
        raise ValueError("Composition, dosage and process inputs must be nonnegative.")
    if (values[ORE] > 100).any().any():
        raise ValueError("Each ore-composition content must be between 0 and 100%.")
    if (values["Liquid_solid"] <= 0).any():
        raise ValueError("Liquid-to-solid ratio must be positive; 0.8 means 0.8:1.")
    return values.astype(float)


def audit_frame(raw: pd.DataFrame, ranges: dict) -> pd.DataFrame:
    outside = []
    for _, row in raw.iterrows():
        outside.append("; ".join(name for name in RAW_FEATURES
                                 if row[name] < ranges[name]["min"] or row[name] > ranges[name]["max"]))
    component_sum = raw[ADDITIVES].sum(axis=1)
    active = (raw[ADDITIVES] > 0).sum(axis=1)
    return pd.DataFrame({
        "additive_system": np.select([active == 0, active == 1], ["No additive", "Single additive"], default="Combined additives"),
        "additive_component_sum": component_sum,
        "total_minus_components": raw["Total_ratio"] - component_sum,
        "outside_observed_range": outside,
    }, index=raw.index)


def predict_frame(frame: pd.DataFrame, models: dict, manifest: dict) -> pd.DataFrame:
    raw = validate_frame(frame)
    result = frame.copy()
    for metal in METALS:
        predicted = np.asarray(models[metal].predict(raw), dtype=float).reshape(-1)
        if len(predicted) != len(raw) or not np.isfinite(predicted).all():
            raise ValueError(f"{metal} model returned invalid predictions.")
        result[f"{metal}_predicted_raw_pct"] = predicted
        result[f"{metal}_predicted_display_pct"] = predicted.clip(0, 100)
    for name, column in audit_frame(raw, manifest["ranges"]).items():
        result[name] = column
    result["model_run_id"] = manifest.get("run_id", "")
    result["dataset_sha256"] = manifest.get("dataset_sha256", "")
    return result


def evidence_table(manifest: dict) -> pd.DataFrame:
    rows = []
    for metal in METALS:
        info = manifest["best_models"][metal]
        train, test, cv = info["train"], info["test"], info["cv"]
        rows.append({
            "Metal": metal, "Selected algorithm": info["algorithm"],
            "Training n": train["n"], "Test n": test["n"],
            "Training R²": train["r2"], "Training RMSE (pp)": train["rmse"],
            "Test R²": test["r2"], "Test RMSE (pp)": test["rmse"],
            "5-fold CV R² (mean)": cv["r2_mean"], "5-fold CV R² (SD)": cv["r2_sd"],
            "5-fold CV RMSE (mean, pp)": cv["rmse_mean"], "5-fold CV RMSE (SD, pp)": cv["rmse_sd"],
        })
    return pd.DataFrame(rows)
