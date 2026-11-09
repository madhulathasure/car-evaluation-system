import joblib
import pandas as pd

from src.config import MODEL_PATH, ENCODER_PATH, COLUMNS, TARGET_COL


class CarEvaluator:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.encoders = joblib.load(ENCODER_PATH)

    def _encode(self, row: dict) -> pd.DataFrame:
        encoded = {}
        for col in COLUMNS:
            if col == TARGET_COL:
                continue
            le = self.encoders[col]
            val = str(row[col])
            if val not in le.classes_:
                raise ValueError(f"Invalid value '{val}' for '{col}'. "
                                 f"Allowed: {list(le.classes_)}")
            encoded[col] = [le.transform([val])[0]]
        return pd.DataFrame(encoded)

    def predict(self, row: dict):
        X = self._encode(row)
        idx = int(self.model.predict(X)[0])
        label = self.encoders[TARGET_COL].inverse_transform([idx])[0]

        proba = None
        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(X)[0]
            proba = {cls: round(float(pr), 4)
                     for cls, pr in zip(self.encoders[TARGET_COL].classes_, p)}
        return label, proba


if __name__ == "__main__":
    ev = CarEvaluator()
    sample = {"buying": "vhigh", "maint": "vhigh", "doors": "2",
              "persons": "2", "lug_boot": "small", "safety": "low"}
    label, proba = ev.predict(sample)
    print(f"Prediction: {label}")
    if proba:
        print("Probabilities:", proba)