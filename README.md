# Lab 2: steel production modelling

Open `lab_2.ipynb` for the completed assignment or `lab_2.html` to read the executed notebook without Jupyter. All required exercises are completed, including Pandera validation, regression comparisons, five-fold CV, hyperparameter search, 30 Optuna trials, MLflow tracking and verified model reload. The optional TabFM bonus is omitted.

The selected model is a tuned Random Forest: training-CV RMSE **811.7**, test RMSE **976.9**, test MAE **451.3 thousand tonnes**, and test R² **0.937**. Linear Regression scores better on this particular test split (RMSE **773.9**); the notebook reports that disagreement and preserves selection by training CV.

## Run again

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). From this folder:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python run_notebook.py
.venv/bin/python -m pytest -q
```

The runner uses the current Python interpreter for a fresh notebook kernel, saves cell outputs, and exports HTML. Re-running adds a new study and new MLflow runs; it overwrites the current model and summary artifacts. Seeds and folds are fixed. Runtime depends on the selected model and machine.

For interactive work, open the notebook in an editor with Jupyter support and select this folder's `.venv/bin/python`. Run with this folder as the working directory.

## Files

- `lab_2.ipynb`: completed notebook with code, explanations, tables and plots.
- `lab_2.html`: readable executed export.
- `data/gist_june_2026.xlsx`: unchanged source workbook from Lab 1.
- `lab_utils.py`: row-wise feature generation and validated batch prediction. Keep this beside the notebook and on the Python import path when loading the model.
- `artifacts/`: saved pipeline, schemas, source/model checksums, metrics, predictions, plots, split manifest, monitoring reference and exported experiment records.
- `mlflow.db` and `mlflow_artifacts/`: local experiment store and logged artifacts.
- `optuna.db`: persistent studies and all trial results.
- `requirements.txt`: exact package versions used for execution.
- `run_notebook.py`: execution and HTML export.
- `test_lab.py`: feature edge cases, schema rejection and saved prediction checks.
- `pyproject.toml`: lint and strict type-check settings. The notebook code was also extracted and checked with these settings. Untyped calls into nbformat, nbconvert and IPython are exempt because those display/execution libraries do not provide typed APIs.

To browse MLflow from this folder:

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1
```

The original MLflow store records absolute local artifact paths. After moving the folder, exported CSVs and `artifacts/` remain readable, but old MLflow artifact links may need path migration. On rerun after moving, the notebook detects the changed location and creates a new experiment with artifact paths under the moved folder. Historical CSV exports remain portable.

## Data and interpretation

Source: Global Energy Monitor, Global Iron and Steel Tracker, June 2026 V1. See the workbook's About and Metadata sheets and the [GIST methodology](https://globalenergymonitor.org/projects/global-iron-steel-tracker/). The unchanged workbook is distributed under Creative Commons Attribution 4.0 International. Attribution: Global Energy Monitor, Global Iron and Steel Tracker, June 2026 (V1) release. See the [source license](https://globalenergymonitor.org/creative-commons-public-license/).

The target is reported 2024 crude steel production, measured in thousand metric tonnes. There are 257 plants with observed labels. The 2025 column has only five reported values. Plant, capacity/status and production tables are joined without multiplying plants. The total production row is used directly; route outputs are not added again. Source `unknown` capacity remains missing; documented structural `N/A` capacity becomes zero. Existing status components with unknown capacities make their aggregate unknown.

The 2026 predictor snapshot follows the 2024 target. This is a retrospective estimation exercise, not a validated forecasting system. Reporting selection, sparse categories, related corporate groups and the small labelled sample limit generalization. Correlations, preprocessing, CV and tuning use only training plants. The test scores requested in early lab sections do not drive model selection. The final candidate is selected by training-CV RMSE, including defaults so tuning cannot force a worse model.

The serving function accepts normalized base inputs, not raw spreadsheet sentinels. See `artifacts/example_inputs.csv` and `artifacts/input_schema.yaml`. It validates before calling the complete feature/preprocessing/model pipeline. Load only trusted joblib artifacts.

## Submission

Group: Adam Zeraiki (B00819260), Lina Oumhand (B00820945), Chris William Karam (B00825083), and Liam Szefner (B00822121). Submitter: Adam Zeraiki.

Repository: https://github.com/Azzbee/aidams-lab2-zeraiki-oumhand-karam-szefner

The repository is public at the submitter's request, although the original assignment requested a private repository. No invitation is needed to read a public repository. The submission email is prepared separately and has not been sent. The personal reflection is in Section 6, Feedback, at the bottom of the notebook; the group should review that draft against its experience.
