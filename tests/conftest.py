import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WAF_DIR = PROJECT_ROOT / "services" / "waf"

if str(WAF_DIR) not in sys.path:
    sys.path.insert(0, str(WAF_DIR))

# waf_tfserving.py checks this file exists at import time
os.environ.setdefault("TOKENIZER_PATH", str(PROJECT_ROOT / "models" / "tokenizer_vocab.json"))
os.environ.setdefault("SOC_ENABLED", "0")
os.environ.setdefault("THRESHOLD", "0.9")
