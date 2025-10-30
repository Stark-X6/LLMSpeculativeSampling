# benchmark/run_experiment.py
import os
import torch
import pandas as pd
from datetime import datetime

from .runner import run_single_experiment
from .runner_support import setup_models
from .analyzer import (
    save_results, summarize_length_stats, summarize_heuristic_gamma,
    plot_latency_vs_gamma, plot_throughput_vs_gamma,
    plot_stage_ratio, plot_sorted_vs_unsorted
)
from .utils_io import ensure_dir, symlink_latest, log_message


def main():
    # ==== 0) 全局配置 ====
    model_pairs = ["bloom", "llama"]
    gammas = [2, 3, 4, 6, 8, 10]           # 启发式 gamma 单独跑
    batch_sizes = [2, 4, 8, 16]
    sorted_flags = [0, 1]                  # 0=不排序, 1=排序
    repeat = 5
    ctx_len = 512
    max_new_tokens = 20

    # ==== 1) 创建结果目录 ====
    root_out = os.path.join("results", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    ensure_dir(root_out)
    symlink_latest(root_out)
    log_message(f"Output directory: {root_out}")

    all_rows = []
    heur_rows = []

    # ==== 2) 外层循环：每种模型对只加载一次 ====
    for model_pair in model_pairs:
        log_message(f"Loading models for {model_pair} ...")
        draft_model, target_model, tokenizer = setup_models(model_pair)
        log_message(f"Models for {model_pair} loaded successfully.")

        # ==== 3) 基础 gamma 实验 ====
        for batch_size in batch_sizes:
            for gamma in gammas:
                for sorted_flag in sorted_flags:
                    row = run_single_experiment(
                        model_pair=model_pair,
                        draft_model=draft_model,
                        target_model=target_model,
                        tokenizer=tokenizer,
                        batch_size=batch_size,
                        gamma=gamma,
                        sorted_flag=sorted_flag,
                        repeat=repeat,
                        ctx_len=ctx_len,
                        max_new_tokens=max_new_tokens
                    )
                    all_rows.append(row)

        # ==== 4) 启发式 gamma 实验（复用同一模型） ====
        for batch_size in batch_sizes:
            for sorted_flag in sorted_flags:
                heur_row = run_single_experiment(
                    model_pair=model_pair,
                    draft_model=draft_model,
                    target_model=target_model,
                    tokenizer=tokenizer,
                    batch_size=batch_size,
                    gamma="heuristic",
                    sorted_flag=sorted_flag,
                    repeat=repeat,
                    ctx_len=ctx_len,
                    max_new_tokens=max_new_tokens
                )
                heur_rows.append(heur_row)

        # ==== 5) 当前模型组跑完，释放显存 ====
        del draft_model, target_model, tokenizer
        torch.cuda.empty_cache()
        log_message(f"Released GPU memory for {model_pair}")

    # ==== 6) 汇总与保存 ====
    df_all = pd.DataFrame(all_rows)
    df_heur = pd.DataFrame(heur_rows)

    save_results(df_all[df_all["model_pair"] == "bloom"], os.path.join(root_out, "bloom.csv"))
    save_results(df_all[df_all["model_pair"] == "llama"], os.path.join(root_out, "llama.csv"))
    summarize_heuristic_gamma(df_heur, os.path.join(root_out, "heuristic_gamma_summary.txt"))
    summarize_length_stats(df_all, os.path.join(root_out, "length_stats_sorted_vs_unsorted.csv"))

    # ==== 7) 绘图 ====
    plots_dir = os.path.join(root_out, "plots")
    ensure_dir(plots_dir)

    plot_latency_vs_gamma(df_all, plots_dir)
    plot_throughput_vs_gamma(df_all, plots_dir)
    plot_stage_ratio(df_all, plots_dir)
    plot_sorted_vs_unsorted(df_all, plots_dir)

    log_message("All done.")


if __name__ == "__main__":
    main()
