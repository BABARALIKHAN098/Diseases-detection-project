# Kidney Disease Detection System

A professional machine learning application for kidney disease detection, featuring a Flask web interface, model training pipelines, and a clean project architecture. This repository includes data preprocessing, model training, inference, and deployment-ready components.

## Key Features

- Predicts kidney disease using machine learning models
- Flask-based web application for user-friendly interaction
- Structured pipeline for data ingestion, transformation, and model training
- Persistent artifacts for model and preprocessor serialization
- Built-in logging and error handling for reliable operation

## Repository Structure

- `mlproject/` - Main application and package code
  - `app.py` - Flask application entry point
  - `application.py` - Runtime launcher for the Flask app
  - `requirements.txt` - Python dependencies
  - `setup.py` - Package configuration
  - `src/` - Project source code package
    - `components/` - Data ingestion, cleaning, transformation, and training
    - `pipeline/` - End-to-end training and prediction workflows
    - `utils.py`, `preprocessor.py`, `logger.py`, `exception.py` - Core utilities
- `img/` - Screenshot assets used in documentation
- `logs/` - Generated application logs
- `PROJECT_STRUCTURE.md` - High-level project structure documentation

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd "Diseases detection system"
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/Scripts/activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r mlproject/requirements.txt
   ```

## Usage

1. Run the Flask application:

   ```bash
   python mlproject/app.py
   ```

2. Open a browser and navigate to:

   ```text
   http://127.0.0.1:5000
   ```

3. Use the web interface to upload input data or enter patient parameters for kidney disease prediction.

## Screenshots

<img src="img/Screenshot 2026-06-25 170728.png" alt="Application screenshot 1" width="600" />

<img src="img/Screenshot 2026-06-25 170819.png" alt="Application screenshot 2" width="600" />

<img src="img/Screenshot 2026-06-25 170838.png" alt="Application screenshot 3" width="600" />

<img src="img/Screenshot 2026-06-25 170904.png" alt="Application screenshot 4" width="600" />

## Architecture

- Data ingestion reads raw dataset files from `notebooks/datasets/`
- Preprocessing and feature scaling are implemented in `src/preprocessor.py`
- Model training logic is centralized in `src/components/model_trainer.py`
- Prediction and inference use the trained model artifact located in `mlproject/artifacts/`
- Flask templates and static assets support the user interface

## Notes

- The model artifacts are stored in `mlproject/artifacts/`.
- Logging is captured under `mlproject/logs/` for runtime diagnostics.
- Existing notebooks support EDA and model training experiments.

## License

This project is provided for educational purposes. Review and apply an appropriate license for production use.
