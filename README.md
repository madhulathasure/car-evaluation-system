# Car Evaluation System

Multi-class classification of car acceptability using the UCI Car Evaluation dataset.

## Setup
```bash
pip install -r requirements.txt
```

## Dataset
Download `car.data` from the UCI repository and place it at `data/raw/car.data`.

## Run Pipeline
```bash
python main.py
```

## Launch Demo
```bash
streamlit run app/streamlit_app.py
```

## Outputs
- `models/best_model.pkl` – tuned best model
- `models/encoders.pkl` – label encoders
- `outputs/metrics.json` – evaluation results + CV scores
- `outputs/model_comparison.csv` – all models compared
- `outputs/confusion_matrices.png` – per-model confusion matrices