#!/usr/bin/env python3
"""Single-config online energy measurement for the spec-decode experiments.

Assumes a `vllm serve` server is ALREADY running. Thin orchestration around Zeus
(zeus -> amdsmi -> AMD hardware energy counter); it does not read the counter
itself.

Closed-loop concurrency (inferenceX style): the client caps in-flight requests at
--max-concurrency C, so the server runs a batch of C. One server (launched with
--max-num-seqs >= max C) serves an entire concurrency sweep; C is set per point on
the client, no per-C relaunch.

Fixed-duration steady-state window (Option A). We poll /metrics to skip ramp, then
measure a fixed T-second slice of steady state, excluding request-generator startup
(idle GPU), ramp-up, and drain:

  * OPEN  window when  completed_requests >= start_frac * N   (ramp done)
  * then measure exactly --window-seconds T, then CLOSE

Drain detection is exact for closed loop: the client sends its LAST request when
completed == N - C, after which the batch drains C -> 0. So a window that closes
with completed_at_close < N - C stayed fully saturated (batch == C). We flag
drain_in_window on completed_at_close >= N - C (or an early bench exit); min/max
running are recorded as corroborating evidence.

At each edge we read the Zeus counter and scrape gen_tokens back-to-back in the
same order, so the two intervals coincide (offsets cancel in the ratio). Everything
reported is WINDOWED. J/tok = dE/dTokens over the identical interval; in steady
state this ratio is independent of the exact edges. Latency stats come from the
bench JSON and are whole-run (labeled _wholerun).

Server-side config (max_num_seqs, num_speculative_tokens, spec on/off) is set on
`vllm serve`, NOT here -- pass the same values as metadata.

Run INSIDE a container with zeus + the vllm CLI + GPU visibility, on the same GPU
the server uses (no restricting HIP_VISIBLE_DEVICES so HIP index == physical).
"""
import argparse
import csv
import json
import os
import subprocess
import time
import urllib.request
from datetime import datetime, timezone

from zeus.monitor import ZeusMonitor

GEN_TOKENS = "vllm:generation_tokens_total"
COMPLETED = "vllm:e2e_request_latency_seconds_count"  # histogram count == completed reqs
RUNNING = "vllm:num_requests_running"                  # gauge
WAITING = "vllm:num_requests_waiting"                  # gauge
# Confirm these two against a server launched WITH speculative_config
# (curl /metrics | grep spec_decode). Absent -> acceptance stays None.
SPEC_ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
SPEC_DRAFT = "vllm:spec_decode_num_draft_tokens_total"
WANTED = (GEN_TOKENS, COMPLETED, RUNNING, WAITING, SPEC_ACCEPTED, SPEC_DRAFT)


def scrape_metrics(base_url):
    """Current value of each wanted series, summed over label sets. Missing -> 0.0."""
    url = base_url.rstrip("/") + "/metrics"
    text = urllib.request.urlopen(url, timeout=10).read().decode()
    out = {k: 0.0 for k in WANTED}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in out:
            try:
                out[name] += float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                pass
    return out


def build_bench_cmd(args, run_dir, filename, num_prompts, save):
    cmd = [
        "vllm", "bench", "serve",
        "--backend", args.backend,
        "--base-url", args.base_url,
        "--endpoint", args.endpoint,
        "--model", args.model,
        "--tokenizer", args.tokenizer,
        "--dataset-name", args.dataset_name,
        "--num-prompts", str(num_prompts),
        "--request-rate", str(args.request_rate),
        "--seed", str(args.seed),
        "--temperature", str(args.temperature),
    ]
    if args.max_concurrency and args.max_concurrency > 0:
        cmd += ["--max-concurrency", str(args.max_concurrency)]
    if args.dataset_name == "spec_bench":
        cmd += ["--dataset-path", args.dataset_path,
                "--spec-bench-output-len", str(args.spec_bench_output_len)]
        if args.spec_bench_category:
            cmd += ["--spec-bench-category", args.spec_bench_category]
    elif args.dataset_name == "random":
        cmd += ["--random-input-len", str(args.random_input_len),
                "--random-output-len", str(args.random_output_len),
                "--ignore-eos"]
    if save:
        cmd += ["--save-result", "--result-dir", run_dir, "--result-filename", filename]
    return cmd


def energy_joules(meas, gpu):
    e = getattr(meas, "gpu_energy", None)
    if e is None:
        e = getattr(meas, "energy", None)
    if isinstance(e, dict):
        return float(e.get(gpu, sum(e.values())))
    return float(getattr(meas, "total_energy", e))


def measure_steady(args, run_dir, mon):
    """Skip ramp, measure a fixed T-second steady window with co-sampled edges.

    Returns (meas, m_start, m_end, diag). Raises RuntimeError if ramp never
    completes (run too short -> raise --num-prompts / lower --steady-start-frac).
    """
    N = args.num_prompts
    target_batch = args.max_concurrency or args.max_num_seqs or 1
    open_at = args.steady_start_frac * N
    drain_at = N - target_batch  # completions at which the batch begins to drain
    T = args.window_seconds

    m_launch = scrape_metrics(args.base_url)  # baseline for this run's completed count
    cmd = build_bench_cmd(args, run_dir, "bench.json", N, save=True)
    print("[measure_one] load:", " ".join(cmd), flush=True)
    logf = open(os.path.join(run_dir, "bench_stdout.log"), "w")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)

    try:
        # Phase 1: wait for ramp to finish
        while True:
            if proc.poll() is not None:
                raise RuntimeError(
                    "bench exited before the steady window opened; raise "
                    "--num-prompts or lower --steady-start-frac")
            m = scrape_metrics(args.base_url)
            completed = m[COMPLETED] - m_launch[COMPLETED]
            if completed >= open_at:
                break
            time.sleep(args.poll_interval)

        # Phase 2: OPEN -- energy read then token scrape (order repeated at close)
        mon.begin_window(args.run_tag)
        m_start = scrape_metrics(args.base_url)
        t_open = time.perf_counter()
        print(f"[steady] OPEN  @ completed={int(completed)}/{N} "
              f"({completed / N:.0%}); batch~{target_batch}; measuring {T:.0f}s", flush=True)

        # Phase 3: measure exactly T seconds; track batch fullness
        min_waiting = m_start[WAITING]
        min_running = max_running = m_start[RUNNING]
        sum_running, n_running = float(m_start[RUNNING]), 1
        early = False
        while True:
            remaining = T - (time.perf_counter() - t_open)
            if remaining <= 0:
                break
            if proc.poll() is not None:
                early = True
                break
            time.sleep(min(args.poll_interval, remaining))
            md = scrape_metrics(args.base_url)
            min_waiting = min(min_waiting, md[WAITING])
            min_running = min(min_running, md[RUNNING])
            max_running = max(max_running, md[RUNNING])
            sum_running += md[RUNNING]
            n_running += 1

        # Phase 4: CLOSE -- same order as open
        meas = mon.end_window(args.run_tag)
        m_end = scrape_metrics(args.base_url)
    finally:
        proc.wait()
        logf.close()

    completed_close = int(m_end[COMPLETED] - m_launch[COMPLETED])
    drain = early or (completed_close >= drain_at)
    print(f"[steady] CLOSE window={meas.time:.2f}s (target {T:.0f}s); "
          f"completed_at_close={completed_close}/{N} (drain begins at {int(drain_at)})"
          f"{'; EARLY EXIT' if early else ''}", flush=True)
    diag = {
        "completed_at_close": completed_close,
        "min_waiting": int(min_waiting),
        "min_running": int(min_running),
        "mean_running": round(sum_running / n_running, 1),
        "max_running": int(max_running),
        "early_exit": early,
        "drain_in_window": bool(drain),
    }
    return meas, m_start, m_end, diag


def main():
    args = parse_args()
    if args.dataset_name == "spec_bench" and not args.dataset_path:
        raise SystemExit(
            "--dataset-path is required for spec_bench; download it once with:\n"
            "  wget -O spec_bench_question.jsonl https://raw.githubusercontent.com/"
            "hemingkx/Spec-Bench/refs/heads/main/data/spec_bench/question.jsonl")
    run_dir = os.path.join(args.result_dir, args.run_tag)
    os.makedirs(run_dir, exist_ok=True)

    if args.warmup_prompts > 0:
        wcmd = build_bench_cmd(args, run_dir, "warmup.json", args.warmup_prompts, save=False)
        print(f"[measure_one] warmup: {args.warmup_prompts} prompts", flush=True)
        subprocess.run(wcmd, check=True,
                       stdout=open(os.path.join(run_dir, "warmup.log"), "w"),
                       stderr=subprocess.STDOUT)

    mon = ZeusMonitor(gpu_indices=[args.gpu])
    meas, m_start, m_end, diag = measure_steady(args, run_dir, mon)

    # ---- windowed energy + tokens (identical co-sampled interval) ----
    energy_j = energy_joules(meas, args.gpu)
    window_s = float(meas.time)
    win_tokens = int(m_end[GEN_TOKENS] - m_start[GEN_TOKENS])
    win_completed = int(m_end[COMPLETED] - m_start[COMPLETED])
    avg_power_w = energy_j / window_s if window_s else None
    j_per_tok = energy_j / win_tokens if win_tokens else None
    win_tok_s = win_tokens / window_s if window_s else None

    # ---- windowed acceptance (only if a spec config is active) ----
    d_acc = m_end[SPEC_ACCEPTED] - m_start[SPEC_ACCEPTED]
    d_draft = m_end[SPEC_DRAFT] - m_start[SPEC_DRAFT]
    acceptance = (d_acc / d_draft) if d_draft > 0 else None
    gamma = args.num_speculative_tokens
    mean_accepted_len = (d_acc * gamma / d_draft) if (gamma and d_draft > 0) else None

    # ---- whole-run latency from the bench JSON (labeled _wholerun) ----
    b = {}
    bench_json = os.path.join(run_dir, "bench.json")
    if os.path.exists(bench_json):
        with open(bench_json) as f:
            b = json.load(f)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_tag": args.run_tag,
        "model": args.model_label,
        "gpu": args.gpu,
        "max_concurrency": args.max_concurrency,
        "max_num_seqs": args.max_num_seqs,
        "num_speculative_tokens": gamma,
        "spec": args.spec,
        "dataset": args.dataset_name,
        "spec_bench_category": args.spec_bench_category or "all",
        "spec_bench_output_len": args.spec_bench_output_len,
        "seed": args.seed,
        "temperature": args.temperature,
        "num_prompts": args.num_prompts,
        "request_rate": args.request_rate,
        "random_input_len": args.random_input_len,
        "random_output_len": args.random_output_len,
        "steady_start_frac": args.steady_start_frac,
        "window_target_s": args.window_seconds,
        "window_s": round(window_s, 3),
        "window_completed_reqs": win_completed,
        "completed_at_close": diag["completed_at_close"],
        "min_running": diag["min_running"],
        "mean_running": diag["mean_running"],
        "max_running": diag["max_running"],
        "min_waiting": diag["min_waiting"],
        "drain_in_window": diag["drain_in_window"],
        "early_exit": diag["early_exit"],
        "gpu_energy_J": round(energy_j, 1),
        "avg_power_W": round(avg_power_w, 1) if avg_power_w else None,
        "window_tokens": win_tokens,
        "J_per_tok": round(j_per_tok, 6) if j_per_tok else None,
        "window_throughput_tok_s": round(win_tok_s, 1) if win_tok_s else None,
        "output_throughput_tok_s_wholerun": b.get("output_throughput"),
        "mean_ttft_ms_wholerun": b.get("mean_ttft_ms"),
        "mean_tpot_ms_wholerun": b.get("mean_tpot_ms"),
        "mean_itl_ms_wholerun": b.get("mean_itl_ms"),
        "spec_accepted_tokens": int(d_acc) if d_draft > 0 else None,
        "spec_draft_tokens": int(d_draft) if d_draft > 0 else None,
        "acceptance_rate": round(acceptance, 4) if acceptance is not None else None,
        "mean_accepted_len": round(mean_accepted_len, 3) if mean_accepted_len is not None else None,
    }

    csv_path = os.path.join(args.result_dir, "measurements.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    with open(os.path.join(run_dir, "row.json"), "w") as f:
        json.dump(row, f, indent=2)

    print("\n=== measure_one result (windowed = fixed-T steady state) ===")
    for k, v in row.items():
        print(f"{k:34s}: {v}")
    if diag["drain_in_window"]:
        print("\nWARNING: window overlapped drain (completed_at_close >= N-C or early "
              "exit) -> left steady state. Raise --num-prompts or lower --window-seconds.")
    print(f"\nappended -> {csv_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8776")
    ap.add_argument("--endpoint", default="/v1/completions")
    ap.add_argument("--backend", default="openai")
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--tokenizer", default="/models/models/gpt-oss-120b")
    ap.add_argument("--gpu", type=int, default=0,
                    help="HIP index to measure; must be the GPU the server uses")
    ap.add_argument("--max-concurrency", type=int, default=0,
                    help="closed-loop in-flight cap (C); the server batch settles at C")
    ap.add_argument("--dataset-name", default="spec_bench")
    ap.add_argument("--dataset-path", default="",
                    help="dataset file path (required for spec_bench: question.jsonl)")
    ap.add_argument("--spec-bench-output-len", type=int, default=256)
    ap.add_argument("--spec-bench-category", default="",
                    help="restrict to one Spec-Bench category; empty = all categories")
    ap.add_argument("--num-prompts", type=int, default=1024)
    ap.add_argument("--request-rate", default="inf")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy; makes spec-on and spec-off outputs identical")
    ap.add_argument("--random-input-len", type=int, default=512)
    ap.add_argument("--random-output-len", type=int, default=256)
    ap.add_argument("--warmup-prompts", type=int, default=16)
    # steady-state window knobs (Option A: fixed duration)
    ap.add_argument("--steady-start-frac", type=float, default=0.10,
                    help="open the window after this fraction of requests complete (skip ramp)")
    ap.add_argument("--window-seconds", type=float, default=15.0,
                    help="measure exactly this many seconds of steady state")
    ap.add_argument("--poll-interval", type=float, default=0.25,
                    help="seconds between /metrics polls")
    # server-side config -- recorded as metadata only (set these on `vllm serve`)
    ap.add_argument("--max-num-seqs", type=int, default=0)
    ap.add_argument("--num-speculative-tokens", type=int, default=0)
    ap.add_argument("--spec", default="off", choices=["on", "off"])
    ap.add_argument("--model-label", default="gpt-oss-120b")
    ap.add_argument("--result-dir", default="results")
    ap.add_argument("--run-tag", default="run")
    return ap.parse_args()


if __name__ == "__main__":
    main()
