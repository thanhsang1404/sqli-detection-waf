# test_datasets.py
import os
import csv
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- CONFIG ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = str(BASE_DIR / "dataset")
MODEL_PATH = str(BASE_DIR / "models" / "sqli_lstm_attention.keras")   # chỉnh nếu bạn dùng .h5 hoặc tên khác
TOKENIZER_PATH = str(BASE_DIR / "models" / "tokenizer.pkl")
MAXLEN = 50

# Make sure dataset folder exists
os.makedirs(DATASET_DIR, exist_ok=True)

# --- load custom objects if present ---
custom_objects = {}
try:
    from train_clean import CustomAttention
    custom_objects["CustomAttention"] = CustomAttention
except Exception:
    # no custom attention or import issue -> ignore
    pass

print("⏳ Loading model & tokenizer ...")
model = load_model(MODEL_PATH, custom_objects=custom_objects)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
print("✅ Loaded model and tokenizer.")

def show_head_with_lines(path, n=10):
    print(f"\n--- Preview first {n} lines of {path} ---")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            print(f"{i:4d}: {line.rstrip()}")
            if i >= n:
                break
    print("--- end preview ---\n")

def safe_read_two_col_csv(path):
    """
    Đọc CSV có 2 cột (query, label). Thử pandas trước, nếu lỗi thì fallback parsing thủ công:
    - Nếu file có header chứa 'query' và 'label', dùng header đó.
    - Fallback parse mỗi dòng bằng rsplit(',',1) để giữ nguyên dấu phẩy trong query.
    Trả về DataFrame với cột 'query' và 'label'.
    """
    # 1) try pandas normally (most cases)
    try:
        df = pd.read_csv(path)
        # Nếu pandas đọc và có 2 cột tên phù hợp -> ok
        cols = [c.lower() for c in df.columns]
        if "query" in cols and "label" in cols:
            # normalize column names
            colmap = {df.columns[i]: ("query" if cols[i]=="query" else ("label" if cols[i]=="label" else df.columns[i]))
                      for i in range(len(df.columns))}
            df = df.rename(columns=colmap)
            return df[["query", "label"]].copy()
        # nếu pandas đọc nhưng không phải 2 cột, tiếp tục để fallback
        if df.shape[1] == 2:
            df.columns = ["query", "label"]
            return df[["query", "label"]].copy()
        # else raise to trigger fallback
        raise ValueError("pandas read but columns not as expected")
    except Exception as e:
        print(f"⚠️ pandas.read_csv failed or inconsistent for '{os.path.basename(path)}': {e}")
        show_head_with_lines(path, n=8)
        print("⏳ Thử parse thủ công bằng fallback (rsplit on last comma)...")

    # 2) fallback manual parsing
    rows = []
    bad_lines = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
        # detect header
        has_header = ("query" in first_line.lower() and "label" in first_line.lower())
        if not has_header:
            # if no header, treat first line as data
            f.seek(0)
        line_no = 0 if has_header else 1
        for raw in f:
            line_no += 1
            s = raw.rstrip("\n")
            if s.strip() == "":
                continue
            # attempt using csv module first (handles quotes)
            try:
                parsed = next(csv.reader([s]))
                if len(parsed) == 2:
                    query, label = parsed[0].strip(), parsed[1].strip()
                    rows.append({"query": query, "label": label})
                    continue
                # else fallthrough to rsplit
            except Exception:
                pass
            # fallback: split by last comma (rsplit)
            parts = s.rsplit(",", 1)
            if len(parts) == 2:
                query, label = parts[0].strip().strip('"').strip(), parts[1].strip().strip('"').strip()
                rows.append({"query": query, "label": label})
            else:
                bad_lines.append((line_no, s))

    if bad_lines:
        print("❗ Một số dòng không parse được (ví dụ tối đa 10 dòng):")
        for ln, txt in bad_lines[:10]:
            print(f"  line {ln}: {txt[:300]}")
        if len(bad_lines) > 10:
            print(f"  ... ({len(bad_lines)} dòng lỗi tổng cộng).")
        print("Bạn nên mở file và bọc trường 'query' bằng dấu kép \"...\" hoặc export CSV với quoting.")

    df2 = pd.DataFrame(rows)
    return df2

def evaluate_dataset(csv_path, save_preds=True, threshold=0.5):
    df = safe_read_two_col_csv(csv_path)
    if df is None or df.shape[0] == 0:
        print(f"⚠️ Không có dòng hợp lệ trong {csv_path}. Bỏ qua.")
        return None

    # ensure strings
    df["query"] = df["query"].astype(str)
    X = pad_sequences(tokenizer.texts_to_sequences(df["query"]), maxlen=MAXLEN, padding="post")
    y_true = np.array([1 if str(l).strip().lower() in ("sqli", "1", "true", "yes") else 0 for l in df["label"]])

    y_prob = model.predict(X, verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["benign", "sqli"], zero_division=0)

    print(f"\n📂 Dataset: {os.path.basename(csv_path)}")
    print(f"✅ Accuracy: {acc:.4f}")
    print("📊 Confusion matrix:")
    print(cm)
    print("🧾 Classification report:")
    print(report)

    if save_preds:
        out = df.copy()
        out["pred_prob"] = y_prob
        out["pred_label"] = ["sqli" if p==1 else "benign" for p in y_pred]
        fname = os.path.join(os.path.dirname(csv_path), f"predictions_{os.path.basename(csv_path)}")
        out.to_csv(fname, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"💾 Saved predictions -> {fname}")

    return {"file": csv_path, "accuracy": acc, "cm": cm, "report": report, "n": len(df)}

def evaluate_all_tests(dir_path=DATASET_DIR, pattern_prefix="test_"):
    fns = sorted([fn for fn in os.listdir(dir_path) if fn.endswith(".csv") and fn.startswith(pattern_prefix)])
    if not fns:
        print(f"Không tìm thấy file test_*.csv trong '{dir_path}'. Thêm file vào thư mục và chạy lại.")
        return {}
    results = {}
    for fn in fns:
        path = os.path.join(dir_path, fn)
        print("\n" + "="*60)
        print(f"⏳ Processing {fn} ...")
        stats = evaluate_dataset(path, save_preds=True)
        if stats:
            results[fn] = stats
    print("\n--- All done ---")
    return results

if __name__ == "__main__":
    evaluate_all_tests()
