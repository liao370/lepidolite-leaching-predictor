# CoreSynergy Lepidolite Li/Rb/Cs leaching predictor

Streamlit application for screening lepidolite roasting and water-leaching conditions.

Live service target: https://lepidolite-leaching-predictor.onrender.com/

Demo account: \`admin\` / \`222333\`

## Data and model revision

- Complete workbook: 944 rows used as the modeling pool.
- Target-specific 80:20 training/test partitions: Li 755/189, Rb 445/112, Cs 433/109.
- Five-fold shuffled cross-validation is performed within each training partition.
- Compared algorithms: LightGBM, random forest, XGBoost, stacking, extremely randomized trees, GBDT and SVR.
- Hyperparameters were selected with Optuna-TPE. The model-selection criterion is mean five-fold CV RMSE; test-set labels are not used for selection.
- XGBoost is selected for Li, Rb and Cs by the CV criterion and then refitted on all available rows for deployment.
- Raw inputs: six ore-composition contents, 19 individual additive-to-ore ratios, total additive-to-ore ratio, roasting temperature/time, liquid-to-solid ratio and water-leaching temperature/time.
- Particle size is a controlled pretreatment condition rather than a predictor; keep feed consistently below 74 μm (200 mesh) when applying the model.

## Local run

\`\`\`bash
pip install -r requirements.txt
streamlit run app.py
\`\`\`

Set \`APP_USERNAME\` and \`APP_PASSWORD\` for a private deployment. The default demo credentials are for demonstration only.

