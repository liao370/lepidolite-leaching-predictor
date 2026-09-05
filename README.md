# CoreSynergy Lepidolite Li/Rb/Cs leaching predictor

Streamlit application for screening lepidolite roasting and water-leaching conditions.

Live service: https://lepidolite-leaching-predictor.onrender.com/

Demo account: `admin` / `222333`

## Data and model revision

- Records 1–896 are the literature modeling domain. The final 48 rows are subsequent experimental validation records and are excluded from fitting, tuning and model selection.
- Target-specific 80:20 training/test partitions: Li 716/180, Rb 407/102 and Cs 395/99.
- Five-fold shuffled cross-validation is performed within each training set.
- Compared algorithms: LightGBM, random forest, XGBoost, stacking, extremely randomized trees, GBDT and SVR.
- Hyperparameters were optimized with Optuna-TPE. Model selection uses training-set cross-validation; test-set and experimental-validation labels are not used for selection.
- XGBoost is selected for Li, Rb and Cs and then refitted on all available literature records for deployment.
- Raw inputs: six ore-composition contents, 19 individual additive-to-ore ratios, total additive-to-ore ratio, roasting temperature/time, liquid-to-solid ratio and water-leaching temperature/time.
- Particle size is a controlled pretreatment condition rather than a predictor; keep feed consistently below 74 μm (200 mesh) when applying the model.

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Set `APP_USERNAME` and `APP_PASSWORD` for a private deployment. The default demo credentials are for demonstration only.

