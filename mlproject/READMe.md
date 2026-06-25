# D-Health AI — Diseases Detection System

A lightweight Flask web application for predicting disease risk from patient data using a trained machine learning pipeline. This repository contains the web UI, prediction pipeline, model helpers, and utilities to generate PDF reports for patient predictions.

## Key features

- Web UI with sign-up/login and dashboard for predictions.
- Predict pipeline integration (`src/pipeline/predict_pipeline.py`).
- Generates PDF reports for patient submissions (saved in `reports/`).
- Lightweight SQLite user & report store (`data/app.db`).

## Screenshots

Dashboard view:

![Dashboard](img/Screenshot%202026-06-25%20170819.png)

Report preview / generated PDF:

![Report](img/Screenshot%202026-06-25%20170904.png)

Landing / Home page:

![Home](img/Screenshot%202026-06-25%20170728.png)

## Quickstart (development)

1. Create and activate a virtual environment (Windows example):

```
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the app from the `mlproject` folder:

```
python app.py
```

4. Open the app at http://127.0.0.1:5000/

## Project structure (important files)

- `app.py` — Flask application and routes.
- `src/pipeline/predict_pipeline.py` — prediction pipeline wrapper used by the app.
- `templates/` — HTML templates for the web UI.
- `static/` — CSS and static files.
- `img/` — screenshots and image assets (referenced above).
- `data/app.db` — SQLite DB (automatically created).
- `reports/` — generated PDF reports (app writes here).

## Notes & troubleshooting

- If PDF generation fails due to missing library, install `reportlab`:

```
pip install reportlab
```

- If the prediction pipeline fails to initialize, ensure the model artifacts the pipeline expects are present and paths referenced by `src/pipeline/predict_pipeline.py` are correct.

- For Windows paths with spaces, run the commands from the `mlproject` folder to avoid path issues.

## License & contact

This project is provided as-is for demonstration and learning. For questions or help, open an issue or contact the maintainer.
