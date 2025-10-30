"""
Plot & summarize benchmark results for BASS experiments.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from .utils_io import ensure_dir, log_message
import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境下也能保存图片

def save_results(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)
    log_message(f"Saved: {path}")


def summarize_length_stats(df_all: pd.DataFrame, out_csv: str):
    # 取各配置的 mean_len/var_len/min_len/max_len 的平均（按 model_pair, sorted 分组）
    g = df_all.groupby(["model_pair", "sorted"]).agg({
        "mean_len": "mean", "var_len": "mean", "min_len": "min", "max_len": "max"
    }).reset_index()
    g.to_csv(out_csv, index=False)
    log_message(f"Saved: {out_csv}")


def summarize_heuristic_gamma(df_heur: pd.DataFrame, out_txt: str):
    # 直接输出文本版
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(df_heur.to_string(index=False))
    log_message(f"Saved: {out_txt}")


# ====== 配色：BASS 论文风格（简洁蓝橙绿等） ======
COLORS = {
    "bs2": "#1f77b4",   # 蓝
    "bs4": "#ff7f0e",   # 橙
    "bs8": "#2ca02c",   # 绿
    "bs16": "#9467bd",  # 紫
    "bloom_sorted": "#1f77b4",
    "bloom_unsorted": "#ff7f0e",
    "llama_sorted": "#2ca02c",
    "llama_unsorted": "#9467bd",
}


def _subplot_pair(figsize=(8, 8)):
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    return fig, axes


def plot_latency_vs_gamma(df_all: pd.DataFrame, plots_dir: str):
    """ 延迟 vs γ；仅 bs=2,4；上Bloom下LLaMA """
    ensure_dir(plots_dir)
    fig, axes = _subplot_pair()

    for i, model_pair in enumerate(["bloom", "llama"]):
        ax = axes[i]
        sub = df_all[(df_all["model_pair"] == model_pair) &
                     (df_all["batch_size"].isin([2, 4])) &
                     (df_all["gamma"] != "heuristic")]
        for bs, color in [(2, COLORS["bs2"]), (4, COLORS["bs4"])]:
            s = sub[sub["batch_size"] == bs]
            # x=gamma, y=latency（ms/token），按sorted分别算平均（我们图里不区分sorted）
            s2 = s.groupby("gamma", as_index=False)["latency"].mean()
            ax.plot(s2["gamma"], s2["latency"], marker="o", color=color, label=f"bs={bs}")
        ax.set_title(f"Latency vs γ ({model_pair})")
        ax.set_ylabel("Latency (ms/token)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("γ")
    out = os.path.join(plots_dir, "latency_vs_gamma.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close(fig)


def plot_throughput_vs_gamma(df_all: pd.DataFrame, plots_dir: str):
    """ 吞吐 vs γ；仅 bs=8,16；上Bloom下LLaMA """
    ensure_dir(plots_dir)
    fig, axes = _subplot_pair()

    for i, model_pair in enumerate(["bloom", "llama"]):
        ax = axes[i]
        sub = df_all[(df_all["model_pair"] == model_pair) &
                     (df_all["batch_size"].isin([8, 16])) &
                     (df_all["gamma"] != "heuristic")]
        for bs, color in [(8, COLORS["bs8"]), (16, COLORS["bs16"])]:
            s = sub[sub["batch_size"] == bs]
            s2 = s.groupby("gamma", as_index=False)["throughput"].mean()
            ax.plot(s2["gamma"], s2["throughput"], marker="o", color=color, label=f"bs={bs}")
        ax.set_title(f"Throughput vs γ ({model_pair})")
        ax.set_ylabel("Throughput (tokens/s)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("γ")
    out = os.path.join(plots_dir, "throughput_vs_gamma.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close(fig)


def plot_stage_ratio(df_all: pd.DataFrame, plots_dir: str):
    """ 阶段占比（固定 bs=8，γ变化；Bloom一张，LLaMA一张）
        - 条形堆叠：先 Draft vs Verify；再 Verify 内部分成 QKV/Attn/FFN（简单画成并列条或第二层堆叠）
        这里实现成：两层并列 —— 每个γ两根条：Draft、Verify（Verify上用hatch标示QKV/Attn/FFN比例提示）
        如需严格堆叠到一根条里，你可以把三个 verify 子比例相加到 verify_ratio 并用 stacked=True。
    """
    ensure_dir(plots_dir)

    for model_pair in ["bloom", "llama"]:
        sub = df_all[(df_all["model_pair"] == model_pair) &
                     (df_all["batch_size"] == 8) &
                     (df_all["gamma"] != "heuristic")]

        g = sub.groupby("gamma", as_index=False).agg({
            "draft_ratio": "mean", "verify_ratio": "mean",
            "qkv_ratio": "mean", "attn_ratio": "mean", "ffn_ratio": "mean"
        })
        x = g["gamma"].tolist()
        draft = g["draft_ratio"].tolist()
        verify = g["verify_ratio"].tolist()

        fig, ax = plt.subplots(figsize=(10, 5))
        width = 0.35
        idx = range(len(x))
        ax.bar([i - width/2 for i in idx], draft, width=width, label="Draft", color=COLORS["bs2"])
        ax.bar([i + width/2 for i in idx], verify, width=width, label="Verify", color=COLORS["bs4"])

        ax.set_title(f"Stage Ratio (bs=8) — {model_pair}")
        ax.set_xlabel("γ")
        ax.set_ylabel("Ratio (elapsed%)")
        ax.set_xticks(list(idx)); ax.set_xticklabels(x)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()

        out = os.path.join(plots_dir, f"stage_ratio_{model_pair}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close(fig)


def plot_sorted_vs_unsorted(df_all: pd.DataFrame, plots_dir: str):
    """
    排序 vs 不排序（延迟用 bs=4，吞吐用 bs=8，γ变化）
    每张图四条线：
        - Bloom (sorted)
        - Bloom (unsorted)
        - LLaMA (sorted)
        - LLaMA (unsorted)
    """
    ensure_dir(plots_dir)

    # 为不同指标设定不同 batch size
    metric_bs_map = {
        "latency": 4,
        "throughput": 8,
    }

    for metric, fname in [
        ("latency", "sorted_vs_unsorted_latency.png"),
        ("throughput", "sorted_vs_unsorted_throughput.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        fixed_bs = metric_bs_map[metric]

        for model_pair, sorted_flag, color_key, label in [
            ("bloom", 1, "bloom_sorted", "Bloom (sorted)"),
            ("bloom", 0, "bloom_unsorted", "Bloom (unsorted)"),
            ("llama", 1, "llama_sorted", "LLaMA (sorted)"),
            ("llama", 0, "llama_unsorted", "LLaMA (unsorted)"),
        ]:
            sub = df_all[
                (df_all["model_pair"] == model_pair)
                & (df_all["batch_size"] == fixed_bs)
                & (df_all["gamma"] != "heuristic")
                & (df_all["sorted"] == sorted_flag)
            ]
            if sub.empty:
                print(f"[WARN] No data for {model_pair}, sorted={sorted_flag}, bs={fixed_bs}")
                continue

            s2 = sub.groupby("gamma", as_index=False)[metric].mean()
            ax.plot(
                s2["gamma"],
                s2[metric],
                marker="o",
                color=COLORS[color_key],
                label=label,
            )

        ax.set_title(
            f"{metric.capitalize()} vs γ (bs={fixed_bs}) — Sorted vs Unsorted"
        )
        ax.set_xlabel("γ")
        ax.set_ylabel(
            "Latency (ms/token)" if metric == "latency" else "Throughput (tokens/s)"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        out = os.path.join(plots_dir, fname)
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close(fig)
