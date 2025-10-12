#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys

try:
    import pandas as pd
except ImportError:
    print("pandas가 필요합니다. 설치: pip install pandas", file=sys.stderr)
    sys.exit(1)

REQUIRED_COLS = [
    "success",
    "spl",
    "episode_time_sec",
    "num_steps",
    "total_distance_m",
    "llm_calls",
    "start_distance_to_goal",  # not used here but kept for completeness
    "final_distance_to_goal",
]

def compute_metrics(df):
    # 필요한 컬럼 확인
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"CSV에 필요한 컬럼이 없습니다: {missing}")

    # 타입 안정화
    for col in ["success","spl","episode_time_sec","num_steps","total_distance_m","llm_calls"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- SR (macro=micro 동일) ---
    sr = df["success"].mean()

    # --- 성공 에피소드 평균 SPL ---
    succ_mask = df["success"] == 1.0
    spl_success_avg = float(df.loc[succ_mask, "spl"].mean()) if succ_mask.any() else 0.0

    # --- 마이크로 방식 합계 ---
    eps = 1e-12
    total_dist  = float(df["total_distance_m"].sum())
    total_time  = float(df["episode_time_sec"].sum())
    total_steps = float(df["num_steps"].sum())
    total_calls = float(df["llm_calls"].sum())

    calls_per_m     = total_calls / max(total_dist, eps)   # 미터 당 LLM 호출 수 (calls/m)
    sec_per_meter   = total_time  / max(total_dist, eps)   # 미터당 연산 시간 (s/m)
    sec_per_step    = total_time  / max(total_steps, eps)  # 스텝 당 연산 시간 (s/step)

    return {
        "SR": sr,
        "SPL_success_avg": spl_success_avg,
        "calls_per_meter": calls_per_m,
        "sec_per_meter": sec_per_meter,
        "sec_per_step": sec_per_step,
        "N_episodes": int(len(df)),
        "N_success": int(df["success"].sum()),
    }

def main():
    ap = argparse.ArgumentParser(description="Compute ObjNav micro metrics from CSV.")
    ap.add_argument("--csv", default="objnav_hm3d.csv", help="CSV file path (default: objnav_hm3d.csv)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    metrics = compute_metrics(df)

    # 출력 포맷
    print(f"# Episodes               : {metrics['N_episodes']}")
    print(f"# Successes              : {metrics['N_success']}")
    print(f"SR                       : {metrics['SR']:.4f}")
    print(f"SPL (성공 에피 평균)     : {metrics['SPL_success_avg']:.4f}")
    print(f"LLM calls per meter      : {metrics['calls_per_meter']:.4f} calls/m")
    print(f"미터당 연산 시간          : {metrics['sec_per_meter']:.5f} s/m")
    print(f"스텝 당 연산 시간         : {metrics['sec_per_step']:.5f} s/step")

if __name__ == "__main__":
    main()
