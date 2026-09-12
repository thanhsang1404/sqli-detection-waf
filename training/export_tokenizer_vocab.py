# export_tokenizer_vocab.py
# Chạy trên máy dev nơi có tokenizer_explicit.pkl (Keras Tokenizer)
# Mục tiêu: xuất word_index + metadata cần thiết ra JSON để WAF không cần Keras khi unpickle.

import pickle, json, os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
IN = str(BASE_DIR / "models" / "tokenizer_explicit.pkl")
OUT = str(BASE_DIR / "models" / "tokenizer_vocab.json")

if not os.path.exists(IN):
    print(f"❌ Không tìm thấy {IN}. Hãy chắc bạn đang chạy script ở thư mục project và file tokenizer_explicit.pkl tồn tại.")
    raise SystemExit(1)

with open(IN, "rb") as f:
    tk = pickle.load(f)

# Lấy những trường cần thiết
data = {
    "word_index": getattr(tk, "word_index", {}),
    # index_word có thể lớn — chuyển key thành str để JSON safe
    "index_word": {str(k): v for k, v in getattr(tk, "index_word", {}).items()} if getattr(tk, "index_word", None) else {},
    "oov_token": getattr(tk, "oov_token", None),
    "lower": getattr(tk, "lower", True),
    "filters": getattr(tk, "filters", None)
}

# Ghi ra json
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ Đã xuất tokenizer vocab -> {OUT}")
print("Sample word_index (first 20):")
for i, (k, v) in enumerate(data["word_index"].items()):
    print(f"  {k} -> {v}")
    if i >= 19:
        break
