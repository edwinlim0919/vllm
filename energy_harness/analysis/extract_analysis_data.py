#!/usr/bin/env python3
"""Build analysis_data.json from the raw energy-sweep measurements.csv.

This is the ONLY data-reduction step between the measured harness output and the
figures. It does not derive J/tok, MFU, ΔJ% etc. — those are computed in
analysis.html's JS from the raw columns below, so the measured-vs-derived line
stays auditable.

FROZEN-DATASET RULE: measurements.csv is append-only (one row per measure_one
call; a re-run appends again). We dedup by run_tag keeping the LAST row, so each
(model, gamma, concurrency) resolves to exactly one measurement.

Raw columns carried through (verbatim from measure_one.py):
  J_per_tok               = gpu_energy_J / window_tokens   (GPU-package J per output tok)
  window_throughput_tok_s = output tokens / 15s steady window
  avg_power_W, acceptance_rate, mean_accepted_len, mean_running

Pactive = active params/token (dense: full; MoE: top-k active). VERIFY against
each model card; used only for the MFU figure. peak_flops = MI300X bf16 peak
(gpt-oss MXFP4 is dequant-emulated -> bf16 is the right denominator).

Usage:  python3 extract_analysis_data.py measurements.csv analysis_data.json
"""
import csv, collections, json, sys

MODELS = ["llama31_8b", "llama33_70b", "gptoss20b", "gptoss120b"]
META = {  # (architecture, active_params_per_token)
    "llama31_8b":  ("dense", 8.03e9),
    "llama33_70b": ("dense", 70.6e9),
    "gptoss20b":   ("MoE",   3.6e9),
    "gptoss120b":  ("MoE",   5.1e9),
}
Cs = [8, 16, 32, 64, 128, 256, 512]
GAMMAS = [0, 1, 2, 3, 4, 5]
PEAK_FLOPS = 1.307e15  # MI300X bf16 peak FLOP/s
FIELDS = {"J": "J_per_tok", "tp": "window_throughput_tok_s", "acc": "acceptance_rate",
          "L": "mean_accepted_len", "pw": "avg_power_W", "mr": "mean_running"}


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main(csv_path, out_path):
    rows = list(csv.DictReader(open(csv_path)))
    last = {}
    for r in rows:
        last[r["run_tag"]] = r          # frozen dedup: last write wins
    rows = list(last.values())

    out = {"Cs": Cs, "gammas": GAMMAS, "peak_flops": PEAK_FLOPS, "models": {}}
    for m in MODELS:
        arch, pact = META[m]
        d = {"arch": arch, "Pactive": pact}
        for key, col in FIELDS.items():
            grid = {}
            for r in rows:
                if r.get("model") != m:
                    continue
                g = int(r["num_speculative_tokens"]); C = int(r["max_concurrency"])
                v = num(r.get(col, ""))
                if v is not None:
                    grid.setdefault(str(g), {})[str(C)] = round(v, 5)
            d[key] = grid
        out["models"][m] = d

    json.dump(out, open(out_path, "w"))
    n = sum(len(out["models"][m]["J"].get(str(g), {})) for m in MODELS for g in GAMMAS)
    print(f"wrote {out_path}: {n} (model,gamma,C) J/tok cells (expect 168)")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "measurements.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "analysis_data.json"
    main(csv_path, out_path)
