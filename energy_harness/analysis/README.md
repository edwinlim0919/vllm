# Energy-sweep analysis (reproducibility)

Figures for the results section: dense-vs-MoE speculative-decoding energy crossover
across the full 168-point sweep (4 models × γ∈{0..5} × C∈{8,16,32,64,128,256,512}).

## Pipeline

```
data/results/measurements.csv          # raw harness output (append-only)
        │  extract_analysis_data.py     # dedup by run_tag (last wins); carry raw columns
        ▼
analysis_data.json                     # frozen dataset the figures read
        │  analysis.html  (JS)          # derives every plotted quantity from raw columns
        ▼
7 figures (rendered in-browser as SVG) + inline descriptions
```

Regenerate the dataset:

```bash
python3 extract_analysis_data.py /path/to/measurements.csv analysis_data.json
```

Then open `analysis.html` (self-contained; the JSON is embedded — re-embed after a
refresh by replacing the `const DATA = {...}` literal with the new file's contents).

## What is measured vs derived

**Measured** (straight from `measure_one.py`, carried verbatim into `analysis_data.json`):
`J_per_tok` (= `gpu_energy_J`/`window_tokens`, GPU-package J per output token over the
15 s steady window), `window_throughput_tok_s`, `avg_power_W`, `acceptance_rate`,
`mean_accepted_len`, `mean_running`.

**Derived in `analysis.html` JS** (so the line is auditable):

| quantity | formula |
|---|---|
| ΔJ% (Fig 1,2) | `(J[γ] − J[γ0]) / J[γ0] × 100` |
| oracle best-γ | `argmin_{γ>0} J[γ][C]` |
| mean tokens/target-pass | `L̄+1 = mean_accepted_len + 1` (accepted drafts + 1 bonus) |
| compute amplification (Fig 5) | `(2γ+1) / (L̄+1)` |
| verify-pass MFU (Fig 5) | `(γ+1)/(L̄+1) · window_throughput · 2·Pactive / peak_flops` |

`Pactive` = active params/token (dense: full; MoE: top-k active — **verify against
each model card**). `peak_flops` = MI300X bf16 (gpt-oss MXFP4 is dequant-emulated on
MI300X → bf16 is the correct denominator; pending exact code-trace).

## Figures
1. Master ΔJ% vs C, oracle best-γ, all 4 models (the crossover).
2. Per-model ΔJ%, every fixed γ + oracle envelope.
3. Raw J/tok(γ,C).
4. Acceptance vs C at fixed γ (flat → rejects the draft-degradation hypothesis).
5. MFU / saturated-compute mechanism (compute amplification + verify-pass MFU).
6. Power vs C (appendix).
7. Mean accepted length / effective speedup.

## Caveats
Single run per point (no error bars). Llama-70B high-C mildly KV-limited (32k cap).
Llama-8B C8/C16 finite-pool drain (effective C < nominal). MFU absolute values are
below theoretical peak (achievable roofline < peak); the cross-model ratio and trend
carry the argument, not the absolute value.
