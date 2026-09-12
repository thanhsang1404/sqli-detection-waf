# train_multi.py
"""
Huấn luyện mô hình SQLi LSTM + Attention nhiều lần (multi-run)
- Dùng lại tokenizer.pkl và các file CSV đã chuẩn bị:
    dataset/train_final.csv
    dataset/val_final.csv
- Mỗi lần chạy (run) sẽ:
    + đặt seed khác nhau
    + build model mới
    + train + lưu model tốt nhất
    + ghi lại kết quả vào models/multi_run_results.csv

Cách chạy:
    (venv-sqli) $ python train_multi.py
"""

import os
import random
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# =========================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
MULTI_RUN_DIR = MODELS_DIR / "multi_runs"

TRAIN_CSV = DATASET_DIR / "train_final.csv"
VAL_CSV = DATASET_DIR / "val_final.csv"
TOKENIZER_PATH = MODELS_DIR / "tokenizer.pkl"

MULTI_RUN_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. HÀM ĐẶT SEED
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# =========================
# 3. LOAD DỮ LIỆU
# =========================
def load_dataset(csv_path: Path):
    """
    Giả định file CSV có 2 cột: query, label
    label: 'sqli' -> 1, 'benign' -> 0
    """
    df = pd.read_csv(csv_path)
    # Chuẩn hoá tên cột nếu cần
    df.columns = [c.strip().lower() for c in df.columns]

    if "query" not in df.columns or "label" not in df.columns:
        raise ValueError(f"CSV {csv_path} phải có cột 'query' và 'label'")

    X_text = df["query"].astype(str).tolist()
    y_label = df["label"].astype(str).tolist()

    y = np.array([1 if lbl.lower() == "sqli" else 0 for lbl in y_label], dtype="int32")
    return X_text, y


# =========================
# 4. LOAD TOKENIZER & TOKENIZE
# =========================
def load_tokenizer(tokenizer_path: Path):
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy tokenizer tại {tokenizer_path}. "
            f"Hãy đảm bảo đã chạy script train_clean.py / notebook để tạo tokenizer.pkl."
        )
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer


def vectorize_texts(tokenizer, texts, maxlen: int):
    seqs = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(seqs, maxlen=maxlen, padding="post", truncating="post")
    return X


# =========================
# 5. LAYER ATTENTION ĐƠN GIẢN
# =========================
class Attention(tf.keras.layers.Layer):
    """
    Attention đơn giản: tính trọng số cho từng timestep từ LSTM output,
    sau đó lấy tổng có trọng số (context vector).
    """

    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.W = Dense(units, activation="tanh")
        self.V = Dense(1)

    def call(self, inputs):
        # inputs: (batch, time, features)
        score = self.V(self.W(inputs))  # (batch, time, 1)
        weights = tf.nn.softmax(score, axis=1)  # (batch, time, 1)
        context = tf.reduce_sum(weights * inputs, axis=1)  # (batch, features)
        return context


# =========================
# 6. BUILD MÔ HÌNH
# =========================
def build_model(vocab_size: int, maxlen: int) -> tf.keras.Model:
    embed_dim = 128
    lstm_units = 64

    inputs = Input(shape=(maxlen,), name="input_ids")
    x = Embedding(
        input_dim=vocab_size + 1,
        output_dim=embed_dim,
        mask_zero=True,
        name="embedding",
    )(inputs)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True), name="bilstm")(x)
    x = Attention(units=64, name="attention")(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="sqli_lstm_attention")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# =========================
# 7. MAIN HUẤN LUYỆN NHIỀU LẦN
# =========================
def main():
    # Cấu hình
    N_RUNS = 5          # số lần train
    MAXLEN = 50         # phải khớp với mô hình/TF-Serving của bạn
    BASE_SEED = 42
    EPOCHS = 10
    BATCH_SIZE = 64

    print("📥 Đang load dữ liệu...")
    X_train_text, y_train = load_dataset(TRAIN_CSV)
    X_val_text, y_val = load_dataset(VAL_CSV)

    print("📥 Đang load tokenizer...")
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    vocab_size = len(tokenizer.word_index)
    print(f"✅ tokenizer word_index size = {vocab_size}")

    print("🔢 Đang vector hoá dữ liệu...")
    X_train = vectorize_texts(tokenizer, X_train_text, maxlen=MAXLEN)
    X_val = vectorize_texts(tokenizer, X_val_text, maxlen=MAXLEN)

    results = []

    for run_idx in range(1, N_RUNS + 1):
        seed = BASE_SEED + run_idx
        print("\n" + "=" * 60)
        print(f"🚀 RUN {run_idx}/{N_RUNS}  (seed = {seed})")
        print("=" * 60)

        set_seed(seed)

        # Build model
        model = build_model(vocab_size=vocab_size, maxlen=MAXLEN)
        model.summary()

        # File lưu model tốt nhất cho run này
        run_model_path = MULTI_RUN_DIR / f"sqli_lstm_attention_run{run_idx}.keras"

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=str(run_model_path),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
            ),
        ]

        history = model.fit(
            X_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=2,
        )

        # Lấy best val_acc & val_loss trong quá trình train
        val_acc_list = history.history.get("val_accuracy", [])
        val_loss_list = history.history.get("val_loss", [])

        best_val_acc = float(max(val_acc_list)) if val_acc_list else None
        best_val_loss = float(min(val_loss_list)) if val_loss_list else None

        # Đánh giá lại trên val
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

        print(
            f"✅ RUN {run_idx} hoàn thành – "
            f"best_val_acc={best_val_acc:.4f}, final_val_acc={val_acc:.4f}"
        )

        results.append(
            {
                "run": run_idx,
                "seed": seed,
                "best_val_accuracy": best_val_acc,
                "best_val_loss": best_val_loss,
                "final_val_accuracy": float(val_acc),
                "final_val_loss": float(val_loss),
                "model_path": str(run_model_path),
            }
        )

    # Lưu kết quả tổng hợp
    results_df = pd.DataFrame(results)
    out_csv = MULTI_RUN_DIR / "multi_run_results.csv"
    results_df.to_csv(out_csv, index=False, encoding="utf-8")
    print("\n📊 Đã lưu kết quả multi-run tại:", out_csv)


if __name__ == "__main__":
    main()
    