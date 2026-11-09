import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

from src.config import (RAW_DATA, PROCESSED_DATA, COLUMNS, TARGET_COL,
                        MODEL_DIR, ENCODER_PATH, TEST_SIZE, RANDOM_STATE)


def load_raw():
    df = pd.read_csv(RAW_DATA, names=COLUMNS)
    # Normalise '5more' / 'more' style values
    df["doors"] = df["doors"].astype(str).str.replace("5more", "5")
    df["persons"] = df["persons"].astype(str).str.replace("more", "6")
    return df


def encode_and_save(df):
    """Label-encode every column and save encoders for inference."""
    encoders = {}
    df_enc = df.copy()
    for col in df.columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col])
        encoders[col] = le

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(encoders, ENCODER_PATH)
    return df_enc, encoders


def preprocess():
    print("[1/3] Loading raw data...")
    df = load_raw()
    print(f"  Rows={len(df)}  Cols={len(df.columns)}")

    print("[2/3] Encoding categorical columns...")
    df_enc, encoders = encode_and_save(df)

    print("[3/3] Saving processed data...")
    os.makedirs(os.path.dirname(PROCESSED_DATA), exist_ok=True)
    df_enc.to_csv(PROCESSED_DATA, index=False)

    print(f"  Class mapping: {dict(enumerate(encoders[TARGET_COL].classes_))}")
    return df_enc, encoders


def get_train_test():
    df = pd.read_csv(PROCESSED_DATA)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE, stratify=y)


if __name__ == "__main__":
    preprocess()