**Options for `compare_fifo_vs_orca.py`**


**Workload**

| Option           | Default                 | Description                                                                                  |
| ---------------- | ----------------------- | -------------------------------------------------------------------------------------------- |
| `--trace_file`   | data/custom_lengths.csv | CSV with num_prefill_tokens / num_decode_tokens columns defining the i/o length distribution |
| `--num_requests` | 256                     | Total number of requests to simulate                                                         |
| `--qps`          | 4.0                     | Arrival rate in requests/second (Poisson process)                                            |
| `--max_tokens`   | 4096                    | Hard cap on prefill+decode per request; longer rows in the CSV are clipped                   |
| `--seed`         | 42                      | RNG seed — same seed guarantees both schedulers see the identical request stream             |

---

**Hardware/Model**

| Option         | Default                  | Description                                                   |
| -------------- | ------------------------ | ------------------------------------------------------------- |
| `--model_name` | meta-llama/Llama-2-7b-hf | Model to simulate (must have profiling data in Vidur's cache) |
| `--device`     | a100                     | GPU device SKU (a40, a100, h100)                              |

---

**Scheduler**

| Option                | Default  | Description                                                                                                                                  |
| --------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `--batch_size_cap`    | 32       | Max concurrent in-flight requests — applies to both schedulers                                                                               |
| `--m_star`            | 0.8      | FIFO only. KV-cache occupancy threshold for admission; a new request is admitted only if its prefill blocks would keep occupancy ≤ threshold |
| `--preemption_policy` | youngest | FIFO only. Victim selection when KV cache is full during decode: youngest (last admitted) or oldest (longest running)                        |

**Output**

| Option          | Default                     | Description                                                                                                        |
| --------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `--output_dir`  | simulator_output/comparison | Base directory; sub-dirs `orca/` and `fifo/` are created inside it, plus `comparison_summary.csv` at the top level |
| `--store_plots` | False                       | If set, writes per-metric PNG plots for each scheduler (requires working kaleido install)                          |
| `--log_level`   | warning                     | Python logging level; use warning to suppress per-event simulator noise, info for progress messages                |
