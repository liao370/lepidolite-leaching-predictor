# Lepidolite Li/Rb/Cs predictor

This Streamlit service is the deployment bundle generated from the complete
lepidolite literature workbook. It retains all 944 populated records and uses
target-specific 80:20 splits, training-only five-fold cross-validation and
Optuna-TPE selection across seven regressors.

## Current model bundle

| Target | Selected model | Training CV R² | Test R² | Deployment fit |
|---|---|---:|---:|---:|
| Li | Stacking | 0.714 | 0.848 | 944 rows |
| Rb | GBDT | 0.813 | 0.882 | 557 rows |
| Cs | XGBoost | 0.823 | 0.819 | 542 rows |

The sidebar intentionally shows only the five-fold training CV R² and the
held-out test R². The model bundle manifest records the full provenance,
feature ranges and hashes.

## Run

```text
pip install -r requirements.txt
streamlit run app.py
```

The Render service uses `render.yaml`; `APP_USERNAME` and `APP_PASSWORD` are
provided as service environment variables. The six independent experimental
records are held out from training and are used only for external validation.
