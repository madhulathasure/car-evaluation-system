from src.preprocess import preprocess
from src.eda import run_eda
from src.train import train

if __name__ == "__main__":
    print("=== STEP 1: EDA ===")
    run_eda()

    print("\n=== STEP 2: Preprocess ===")
    preprocess()

    print("\n=== STEP 3: Train & Evaluate ===")
    train()

    print("\nDone. Launch demo: streamlit run app/streamlit_app.py")