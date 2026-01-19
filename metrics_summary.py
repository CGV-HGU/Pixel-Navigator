#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys

try:
    import pandas as pd
except ImportError:
    print("pandas가 필요합니다. 설치: pip install pandas", file=sys.stderr)
    sys.exit(1)

import numpy as np  # ✅ NEW: inf/NaN 처리를 위해 추가

REQUIRED_COLS = [
    "success",
    "spl",
    "episode_time_sec",
    "num_steps",
    "total_distance_m",
    "llm_calls",
    "start_distance_to_goal",
    "final_distance_to_goal",
]

def compute_metrics(df):
    # --- 필수 컬럼 체크 ---
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"CSV에 필요한 컬럼이 없습니다: {missing}")

    # --- 타입 안정화 ---
    for col in [
        "success",
        "spl",
        "episode_time_sec",
        "num_steps",
        "total_distance_m",
        "llm_calls",
        "start_distance_to_goal",
        "final_distance_to_goal",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ✅ NEW: ±inf 를 NaN으로 치환 (거리/시간 계산 안정화)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ✅ NEW: SPL NaN 은 0.0으로 처리 (주로 실패/도달불가 에피소드)
    df["spl"] = df["spl"].fillna(0.0)

    # --- 기본 통계 ---
    n_eps       = int(len(df))
    sr          = float(df["success"].mean())

    succ_mask   = df["success"] == 1.0
    spl_success_avg = float(df.loc[succ_mask, "spl"].mean()) if succ_mask.any() else 0.0
    spl_overall = float(df["spl"].mean())
    mean_episode_time_sec = float(df["episode_time_sec"].mean()) if n_eps > 0 else 0.0

    # ✅ 에피소드 평균 이동거리 / 평균 start distance
    mean_total_distance_m   = float(df["total_distance_m"].mean()) if n_eps > 0 else 0.0
    mean_start_distance_m   = float(df["start_distance_to_goal"].mean()) if n_eps > 0 else 0.0

    # ---합계(마이크로용)---
    eps = 1e-12
    total_dist   = float(df["total_distance_m"].sum())
    total_time   = float(df["episode_time_sec"].sum())
    total_steps  = float(df["num_steps"].sum())
    total_calls  = float(df["llm_calls"].sum())

    # --- Micro (총합/총합) ---
    calls_per_meter_micro = total_calls / max(total_dist, eps)
    calls_per_step_micro  = total_calls / max(total_steps, eps)
    sec_per_meter_micro   = total_time  / max(total_dist, eps)
    sec_per_step_micro    = total_time  / max(total_steps, eps)

    # --- Macro (에피소드별 비율의 평균; 0 분모 에피 제외) ---
    dist_pos  = df["total_distance_m"] > 0
    steps_pos = df["num_steps"]         > 0

    calls_per_meter_macro = float(
        (df.loc[dist_pos, "llm_calls"] / df.loc[dist_pos, "total_distance_m"]).mean()
    ) if dist_pos.any() else 0.0
    calls_per_step_macro  = float(
        (df.loc[steps_pos, "llm_calls"] / df.loc[steps_pos, "num_steps"]).mean()
    ) if steps_pos.any() else 0.0
    sec_per_meter_macro   = float(
        (df.loc[dist_pos, "episode_time_sec"] / df.loc[dist_pos, "total_distance_m"]).mean()
    ) if dist_pos.any() else 0.0
    sec_per_step_macro    = float(
        (df.loc[steps_pos, "episode_time_sec"] / df.loc[steps_pos, "num_steps"]).mean()
    ) if steps_pos.any() else 0.0

    # --- 에피소드 평균 호출수(매크로=마이크로 동일) ---
    llm_calls_per_episode = total_calls / max(n_eps, 1)

    return {
        # 규모/성공도
        "N_episodes": n_eps,
        "N_success": int(df["success"].sum()),
        "SR": sr,
        "SPL_success_avg": spl_success_avg,
        "SPL_overall": spl_overall,

        # VLM 호출 효율 (micro / macro)
        "calls_per_meter_micro": calls_per_meter_micro,
        "calls_per_meter_macro": calls_per_meter_macro,
        "calls_per_step_micro":  calls_per_step_micro,
        "calls_per_step_macro":  calls_per_step_macro,

        # 연산 시간 효율 (micro / macro)
        "sec_per_meter_micro": sec_per_meter_micro,
        "sec_per_meter_macro": sec_per_meter_macro,
        "sec_per_step_micro":  sec_per_step_micro,
        "sec_per_step_macro":  sec_per_step_macro,

        # 에피소드당 평균 호출수
        "llm_calls_per_episode": llm_calls_per_episode,
        # 에피소드당 평균 소요시간
        "mean_episode_time_sec": mean_episode_time_sec,

        # 평균 이동거리 / 평균 start distance
        "mean_total_distance_m": mean_total_distance_m,
        "mean_start_distance_m": mean_start_distance_m,
    }

def main():
    ap = argparse.ArgumentParser(description="Compute ObjNav metrics (micro & macro) from CSV.")
    ap.add_argument("--csv", default="objnav_hm3d.csv", help="CSV file path (default: objnav_hm3d.csv)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    m = compute_metrics(df)

    print(f"# Episodes                   : {m['N_episodes']}")
    print(f"# Successes                  : {m['N_success']}")
    print(f"SR                           : {m['SR']:.4f}")
    #print(f"SPL (성공 에피 평균/macro)   : {m['SPL_success_avg']:.4f}")
    print(f"SPL (전체 평균/macro)        : {m['SPL_overall']:.4f}")

    print("\n-- VLM 호출 효율 --")
    print(f"LLM calls per meter (micro)  : {m['calls_per_meter_micro']:.3f} calls/m")
    #print(f"LLM calls per meter (macro)  : {m['calls_per_meter_macro']:.4f} calls/m")
    #print(f"LLM calls per step  (micro)  : {m['calls_per_step_micro']:.4f} calls/step")
    #print(f"LLM calls per step  (macro)  : {m['calls_per_step_macro']:.4f} calls/step")
    print(f"LLM calls per episode        : {m['llm_calls_per_episode']:.3f} calls/ep")

    print("\n-- 연산 시간 효율 --")
    print(f"sec per meter (micro)        : {m['sec_per_meter_micro']:.4f} s/m")
    #print(f"sec per meter (macro)        : {m['sec_per_meter_macro']:.4f} s/m")
    print(f"sec per step  (micro)        : {m['sec_per_step_micro']:.4f} s/step")
    #print(f"sec per step  (macro)        : {m['sec_per_step_macro']:.4f} s/step")
    print(f"avg episode time (macro)     : {m['mean_episode_time_sec']:.2f} s/ep")

    print("\n-- 거리 통계 --")
    print(f"avg traveled distance        : {m['mean_total_distance_m']:.2f} m/ep")
    print(f"avg start dist to goal       : {m['mean_start_distance_m']:.2f} m")

if __name__ == "__main__":
    main()
