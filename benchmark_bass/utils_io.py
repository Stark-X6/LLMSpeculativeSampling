# benchmark_bass/utils_io.py
import os
import json
import shutil
from datetime import datetime


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def log_message(msg: str, level: str = "INFO"):
    print(f"[{level}] {msg}")


def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_csv(path: str):
    import pandas as pd
    return pd.read_csv(path)


def make_fresh_outdir(base="results"):
    outdir = os.path.join(base, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    ensure_dir(outdir)
    return outdir


def symlink_latest(target_dir: str):
    latest = os.path.join(os.path.dirname(target_dir), "latest")
    try:
        if os.path.islink(latest) or os.path.exists(latest):
            try:
                os.remove(latest)
            except IsADirectoryError:
                shutil.rmtree(latest)
        os.symlink(os.path.abspath(target_dir), latest, target_is_directory=True)
    except Exception as e:
        # 某些文件系统不支持symlink，忽略
        log_message(f"symlink_latest failed: {e}", level="WARN")
