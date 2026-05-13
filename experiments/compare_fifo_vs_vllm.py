"""
Compare the FIFO and vLLM replica schedulers under identical workloads.

Usage
-----
python experiments/compare_fifo_vs_vllm.py \
    --trace_file data/custom_lengths.csv \
    --num_requests 512 \
    --qps 16.0 \
    --batch_size_cap 128 \
    --max_tokens 8192 \
    --m_star 0.95
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from vidur.entities.batch import Batch
from vidur.entities.batch_stage import BatchStage
from vidur.entities.cluster import Cluster
from vidur.entities.replica import Replica
from vidur.entities.request import Request

from vidur.config import (
    ClusterConfig,
    CustomRequestLengthGeneratorConfig,
    FifoSchedulerConfig,
    MetricsConfig,
    PoissonRequestIntervalGeneratorConfig,
    RandomForrestExecutionTimePredictorConfig,
    ReplicaConfig,
    SimulationConfig,
    SyntheticRequestGeneratorConfig,
    VllmSchedulerConfig,
)
from vidur.metrics.constants import (
    BatchMetricsCountDistribution,
    RequestMetricsHistogram,
    RequestMetricsTimeDistributions,
    TokenMetricsTimeDistribution,
)
from vidur.simulator import Simulator
from vidur.utils.random import set_seeds


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _make_config(
    scheduler_config,
    args: argparse.Namespace,
    label: str,
) -> SimulationConfig:
    length_config = CustomRequestLengthGeneratorConfig(
        trace_file=args.trace_file,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    interval_config = PoissonRequestIntervalGeneratorConfig(
        qps=args.qps,
    )

    request_gen_config = SyntheticRequestGeneratorConfig(
        length_generator_config=length_config,
        interval_generator_config=interval_config,
        num_requests=args.num_requests,
        seed=args.seed,
    )

    replica_config = ReplicaConfig(
        model_name=args.model_name,
        device=args.device,
        tensor_parallel_size=1,
        num_pipeline_stages=1,
    )

    cluster_config = ClusterConfig(
        num_replicas=1,
        replica_config=replica_config,
        replica_scheduler_config=scheduler_config,
    )

    metrics_config = MetricsConfig(
        output_dir=f"{args.output_dir}/{label}",
        write_metrics=True,
        store_plots=args.store_plots,
        enable_chrome_trace=False,
        write_json_trace=False,
        store_token_completion_metrics=True,
    )

    predictor_config = RandomForrestExecutionTimePredictorConfig()

    return SimulationConfig(
        seed=args.seed,
        log_level=args.log_level,
        cluster_config=cluster_config,
        request_generator_config=request_gen_config,
        execution_time_predictor_config=predictor_config,
        metrics_config=metrics_config,
    )


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def _ds_stats(data_series, col_name: str) -> dict:
    df = data_series._to_df()
    if len(df) == 0:
        return {k: float("nan") for k in ("mean", "p50", "p95", "p99")}

    s = df[col_name]
    return {
        "mean": s.mean(),
        "p50": s.quantile(0.50),
        "p95": s.quantile(0.95),
        "p99": s.quantile(0.99),
    }


def _cdf_stats(cdf_sketch) -> dict:
    sk = cdf_sketch._sketch
    if sk._count == 0:
        return {k: float("nan") for k in ("mean", "p50", "p95", "p99")}

    return {
        "mean": sk.avg,
        "p50": sk.get_quantile_value(0.50),
        "p95": sk.get_quantile_value(0.95),
        "p99": sk.get_quantile_value(0.99),
    }


def _reset_entity_ids() -> None:
    """
    Reset class-level ID counters so each simulation run starts from scratch.

    Without this, the second simulator run may create Replica(id=1), while
    Vidur's global scheduler still expects replica_id=0.
    """
    for cls in (Replica, Cluster, Batch, BatchStage, Request):
        cls._id = -1


def extract_metrics(sim: Simulator) -> dict:
    ms = sim.metric_store

    e2e = _ds_stats(
        ms._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_E2E_TIME
        ],
        RequestMetricsTimeDistributions.REQUEST_E2E_TIME.value,
    )

    ttft = _ds_stats(
        ms._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.PREFILL_TIME_E2E
        ],
        RequestMetricsTimeDistributions.PREFILL_TIME_E2E.value,
    )

    sched_delay = _ds_stats(
        ms._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_SCHEDULING_DELAY
        ],
        RequestMetricsTimeDistributions.REQUEST_SCHEDULING_DELAY.value,
    )

    tpot = _cdf_stats(
        ms._token_metrics_time_distribution[
            TokenMetricsTimeDistribution.DECODE_TOKEN_EXECUTION_PLUS_PREMPTION_TIME
        ]
    )

    batch_size = _cdf_stats(
        ms._batch_metrics_count_distribution[
            BatchMetricsCountDistribution.BATCH_SIZE
        ]
    )

    restarts_ds = ms._request_metrics_histogram[
        RequestMetricsHistogram.REQUEST_NUM_RESTARTS
    ]
    restarts_df = restarts_ds._to_df()
    restart_col = RequestMetricsHistogram.REQUEST_NUM_RESTARTS.value

    total_restarts = int(restarts_df[restart_col].sum()) if len(restarts_df) else 0
    mean_restarts = restarts_df[restart_col].mean() if len(restarts_df) else 0.0

    return {
        "e2e_mean_s": e2e["mean"],
        "e2e_p50_s": e2e["p50"],
        "e2e_p95_s": e2e["p95"],
        "e2e_p99_s": e2e["p99"],
        "ttft_mean_s": ttft["mean"],
        "ttft_p50_s": ttft["p50"],
        "ttft_p95_s": ttft["p95"],
        "ttft_p99_s": ttft["p99"],
        "tpot_mean_s": tpot["mean"],
        "tpot_p50_s": tpot["p50"],
        "tpot_p95_s": tpot["p95"],
        "tpot_p99_s": tpot["p99"],
        "sched_delay_mean_s": sched_delay["mean"],
        "sched_delay_p99_s": sched_delay["p99"],
        "batch_size_mean": batch_size["mean"],
        "batch_size_p99": batch_size["p99"],
        "total_restarts": total_restarts,
        "mean_restarts_per_req": mean_restarts,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "e2e_mean_s": "E2E latency        mean (s)",
    "e2e_p50_s": "E2E latency         p50 (s)",
    "e2e_p95_s": "E2E latency         p95 (s)",
    "e2e_p99_s": "E2E latency         p99 (s)",
    "ttft_mean_s": "TTFT               mean (s)",
    "ttft_p50_s": "TTFT                p50 (s)",
    "ttft_p95_s": "TTFT                p95 (s)",
    "ttft_p99_s": "TTFT                p99 (s)",
    "tpot_mean_s": "TPOT               mean (s)",
    "tpot_p50_s": "TPOT                p50 (s)",
    "tpot_p95_s": "TPOT                p95 (s)",
    "tpot_p99_s": "TPOT                p99 (s)",
    "sched_delay_mean_s": "Sched delay        mean (s)",
    "sched_delay_p99_s": "Sched delay         p99 (s)",
    "batch_size_mean": "Batch size             mean",
    "batch_size_p99": "Batch size              p99",
    "total_restarts": "Total restarts / preemptions",
    "mean_restarts_per_req": "Mean restarts / request",
}


def print_comparison(vllm_metrics: dict, fifo_metrics: dict) -> None:
    col_w = 14
    label_w = 36

    header = (
        f"{'Metric':<{label_w}}  "
        f"{'vLLM':>{col_w}}  "
        f"{'FIFO':>{col_w}}  "
        f"{'FIFO / vLLM':>{col_w}}"
    )
    sep = "-" * len(header)

    print()
    print(sep)
    print(header)
    print(sep)

    for key, label in METRIC_LABELS.items():
        vllm_val = vllm_metrics[key]
        fifo_val = fifo_metrics[key]
        ratio = fifo_val / vllm_val if vllm_val != 0 else float("nan")

        if isinstance(vllm_val, float) or isinstance(fifo_val, float):
            print(
                f"{label:<{label_w}}  "
                f"{vllm_val:>{col_w}.4f}  "
                f"{fifo_val:>{col_w}.4f}  "
                f"{ratio:>{col_w}.4f}"
            )
        else:
            print(
                f"{label:<{label_w}}  "
                f"{vllm_val:>{col_w}}  "
                f"{fifo_val:>{col_w}}  "
                f"{ratio:>{col_w}.4f}"
            )

    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare FIFO vs vLLM scheduler under identical workloads."
    )

    # Workload
    p.add_argument(
        "--trace_file",
        default="data/custom_lengths.csv",
        help="CSV with num_prefill_tokens and num_decode_tokens columns.",
    )
    p.add_argument("--num_requests", type=int, default=512)
    p.add_argument(
        "--qps",
        type=float,
        default=16.0,
        help="Arrival rate in requests/second.",
    )
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--seed", type=int, default=42)

    # Hardware / model
    p.add_argument("--model_name", default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--device", default="a100")

    # Scheduler
    p.add_argument(
        "--batch_size_cap",
        type=int,
        default=128,
        help="Maximum concurrent in-flight requests.",
    )

    # FIFO-specific
    p.add_argument(
        "--m_star",
        type=float,
        default=0.95,
        help="FIFO KV-cache occupancy admission threshold.",
    )
    p.add_argument(
        "--preemption_policy",
        default="youngest",
        choices=["youngest", "oldest"],
        help="FIFO victim selection policy when KV cache is full.",
    )

    # Output
    p.add_argument(
        "--output_dir",
        default="simulator_output/comparison_fifo_vs_vllm",
    )
    p.add_argument("--store_plots", action="store_true", default=False)
    p.add_argument("--log_level", default="warning")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    vllm_sched = VllmSchedulerConfig(
        batch_size_cap=args.batch_size_cap,
    )

    fifo_sched = FifoSchedulerConfig(
        batch_size_cap=args.batch_size_cap,
        m_star=args.m_star,
        preemption_policy=args.preemption_policy,
    )

    runs = [
        ("vllm", vllm_sched),
        ("fifo", fifo_sched),
    ]

    results = {}

    for label, sched in runs:
        print(f"[{label.upper()}] Running simulation ...")

        _reset_entity_ids()
        set_seeds(args.seed)

        config = _make_config(sched, args, label)
        sim = Simulator(config)
        sim.run()

        results[label] = extract_metrics(sim)

        print(f"[{label.upper()}] Done. Output: {config.metrics_config.output_dir}")

    print_comparison(results["vllm"], results["fifo"])

    df = pd.DataFrame(results).rename(
        columns={
            "vllm": "vLLM",
            "fifo": "FIFO",
        }
    )

    df.index = [METRIC_LABELS.get(k, k) for k in df.index]
    df["FIFO / vLLM"] = df["FIFO"] / df["vLLM"]

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = f"{args.output_dir}/comparison_summary.csv"
    df.to_csv(out_csv)

    print(f"Summary saved to {out_csv}")


if __name__ == "__main__":
    main()