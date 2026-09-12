# =========================================
# SQL Injection Detection with LSTM + Custom Attention
# Giữ mask_zero=True, tránh lỗi Dimension mismatch
# =========================================

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Layer, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
import pickle, os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ==== 1. Dataset mẫu (bạn có thể mở rộng sau) ====
data = {
    "query": [
        "SELECT * FROM users",
        "SELECT id FROM products WHERE id=1",
        "'; DROP TABLE users; --",
        "1 OR 1=1",
        "admin' OR '1'='1",
        "INSERT INTO accounts VALUES ('test','123')",
        "UPDATE users SET password='123' WHERE id=1",
        "SELECT name FROM users WHERE username='admin' AND password='123' --"
    ],
    "label": ["benign", "benign", "sqli", "sqli", "sqli", "sqli", "benign", "sqli"]
}
df = pd.DataFrame(data)

# ==== 2. Tokenization ====
queries = df["query"].astype(str).str.lower().tolist()
labels = np.array([1 if x == "sqli" else 0 for x in df["label"]])

tokenizer = Tokenizer(num_words=2000, oov_token="<OOV>")
tokenizer.fit_on_texts(queries)
sequences = tokenizer.texts_to_sequences(queries)
MAXLEN = 50
X = pad_sequences(sequences, maxlen=MAXLEN, padding='post')

# ==== 3. Custom Attention Layer ====
class CustomAttention(Layer):
    def __init__(self, **kwargs):
        super(CustomAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight",
                                 shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias",
                                 shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(CustomAttention, self).build(input_shape)

    def call(self, x, mask=None):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        e = tf.keras.backend.squeeze(e, axis=-1)

        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            e -= (1.0 - mask) * 1e9  # loại bỏ padding positions

        a = tf.keras.backend.softmax(e)
        a = tf.keras.backend.expand_dims(a)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

# ==== 4. Xây dựng Model ====
inputs = Input(shape=(MAXLEN,), name="input_layer")
x = Embedding(input_dim=2000, output_dim=64, mask_zero=True, name="embedding")(inputs)
lstm_out = LSTM(64, return_sequences=True, name="lstm")(x)
att_out = CustomAttention(name="attention")(lstm_out)
outputs = Dense(1, activation='sigmoid', name="output")(att_out)

model = Model(inputs, outputs)
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# ==== 5. Huấn luyện ====
history = model.fit(X, labels, epochs=10, batch_size=2, validation_split=0.2, verbose=1)

# ==== 6. Lưu model và tokenizer ====
os.makedirs(str(BASE_DIR / "models"), exist_ok=True)
model.save(str(BASE_DIR / "models" / "sqli_lstm_attention.keras"))

with open(str(BASE_DIR / "models" / "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Model và tokenizer đã được lưu vào /models")
