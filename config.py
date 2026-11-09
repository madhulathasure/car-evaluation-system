import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", "..", "..", "..", ".."))

RAW_DATA = os.path.join(PROJECT_ROOT, "data", "raw", "car.data")
PROCESSED_DATA = os.path.join(PROJECT_ROOT, "data", "raw", "processed", "car_clean.csv")

MODEL_DIR = os.path.join(SRC_DIR, "models")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoders.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")

OUTPUT_DIR = os.path.join(SRC_DIR, "models", "app")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")
COMPARISON_PATH = os.path.join(OUTPUT_DIR, "model_comparison.csv")
CONFUSION_PATH = os.path.join(OUTPUT_DIR, "confusion_matrices.png")

COLUMNS = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
TARGET_COL = "class"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5