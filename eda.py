import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.config import RAW_DATA, COLUMNS, TARGET_COL, OUTPUT_DIR


def run_eda():
    df = pd.read_csv(RAW_DATA, names=COLUMNS)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Class distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x=TARGET_COL, data=df, palette="viridis")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"))
    plt.close()

    # 2. Feature vs class grids
    features = [c for c in COLUMNS if c != TARGET_COL]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, feat in zip(axes.flatten(), features):
        sns.countplot(x=feat, hue=TARGET_COL, data=df, ax=ax, palette="Set2")
        ax.set_title(f"{feat} vs {TARGET_COL}")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "features_vs_class.png"))
    plt.close()

    print("EDA plots saved to outputs/")


if __name__ == "__main__":
    run_eda()