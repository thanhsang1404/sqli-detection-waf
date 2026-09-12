# evaluate_new_dataset.py
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
import argparse

# ==========================
# ⚙️ CLI arguments
# ==========================
parser = argparse.ArgumentParser(description="Evaluate trained SQLi model on new dataset")
parser.add_argument("--data", required=True, help="Path to CSV dataset file (must contain 'query' and 'label')")
args = parser.parse_args()

# ==========================
# 📂 Paths
# ==========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / args.data
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "sqli_lstm_attention_explicit.keras"
TOKENIZER_PATH = MODEL_DIR / "tokenizer_explicit.pkl"
MAXLEN = 50  # same as training

# ==========================
# 🔹 Load model & tokenizer
# ==========================
print("Loading model and tokenizer...")
model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# ==========================
# 📊 Load dataset
# ==========================
df = pd.read_csv(DATA_PATH)
if not {"query", "label"}.issubset(df.columns):
    raise ValueError("❌ Dataset must contain columns: query, label")

X_test = pad_sequences(tokenizer.texts_to_sequences(df["query"].astype(str)),
                       maxlen=MAXLEN, padding="post")
y_test = np.array(df["label"].astype(str).str.lower().map(lambda x: 1 if x == "sqli" else 0))

print(f"✅ Loaded dataset: {len(df)} samples from {DATA_PATH.name}")

# ==========================
# 🤖 Predict
# ==========================
y_prob = model.predict(X_test, verbose=0).reshape(-1)
y_pred = (y_prob >= 0.5).astype(int)

# ==========================
# 📈 Evaluation report
# ==========================
print("\n=== Evaluation Results ===")
print(classification_report(y_test, y_pred, target_names=["benign", "sqli"], digits=3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ==========================
# 💾 (Optional) Save predictions
# ==========================
out_path = MODEL_DIR / "predictions.csv"
df["predicted"] = ["sqli" if p == 1 else "benign" for p in y_pred]
df["probability"] = y_prob
df.to_csv(out_path, index=False)
print(f"\n📁 Predictions saved to: {out_path}")
