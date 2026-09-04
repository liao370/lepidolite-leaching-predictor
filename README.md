# Lepidolite Li/Rb/Cs Leaching Predictor

Public Streamlit application for screening lepidolite roasting–water-leaching conditions.

Live site: https://lepidolite-leaching-predictor.onrender.com/

Demo login: `admin` / `222333`

## Model revision (2026-09-04)

- Literature dataset: 896 usable records.
- Target-specific sample sizes: Li 896, Rb 509 and Cs 494.
- Validation design: random 80:20 holdout (seed 492), with shuffled five-fold cross-validation inside the training set.
- Hyperparameter search: Optuna TPE.
- Compared algorithms: LightGBM, Random Forest, XGBoost, Stacking, Extra Trees, GBDT and SVR.
- Selection rule: the one-standard-error rule followed by parsimony, using literature-training cross-validation only. Chemistry-informed XGBoost was selected for all three targets.
- Independent validation: 46 experiments selected by a response-blind exact-replicate cap; all 48 experiments are retained as a sensitivity analysis. Experimental labels were never used for fitting, tuning or selection.
- Literature holdout R²: Li 0.744, Rb 0.860 and Cs 0.880.
- Independent 46-experiment R²: Li 0.208, Rb 0.407 and Cs 0.125.

## Input schema

The previous binary grade input has been removed. The application now uses all 31 predictors:

- 6 ore-composition descriptors: Li2O, Rb, Cs, SiO2, Al2O3 and Fe2O3 contents;
- 19 individual additive-to-ore ratios;
- total additive-to-ore ratio;
- roasting temperature and time;
- liquid-to-solid ratio;
- water-leaching temperature and time.

Particle size is treated as a controlled pretreatment condition rather than a model predictor. For comparability with the training literature, samples should be ground to <74 µm.

## Interpretation

The interface keeps all 31 raw inputs while the fitted models additionally derive reagent-family totals, component fractions, composition ratios and process-intensity descriptors. Displayed predictions are constrained to the physical interval 0–100%. The application is intended for screening candidate conditions; final process performance should be confirmed experimentally on the same ore batch.

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The public shared credentials are the defaults. For a private deployment, set `APP_USERNAME` and `APP_PASSWORD` as environment variables or configure `.streamlit/secrets.toml`.
