# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Vidur

Vidur is a high-fidelity LLM inference system **simulator** — it models the performance of LLM serving systems without requiring GPU access (except for initial profiling). It supports various schedulers (Sarathi, vLLM, Orca, FasterTransformer, LightLLM), devices (A40, A100, H100), and parallelism strategies (tensor/pipeline parallel).

## Commands

### Setup (Python 3.10+ required)
```bash
# Recommended
mamba env create -p ./env -f ./environment.yml
mamba env update -f environment-dev.yml

# Alternative (venv)
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run the simulator
```bash
python -m vidur.main --help

# Example run
python -m vidur.main \
  --replica_config_device a100 \
  --replica_config_model_name meta-llama/Meta-Llama-3-8B \
  --cluster_config_num_replicas 1 \
  --replica_config_tensor_parallel_size 1 \
  --replica_config_num_pipeline_stages 1 \
  --request_generator_config_type synthetic \
  --synthetic_request_generator_config_num_requests 512 \
  --length_generator_config_type trace \
  --interval_generator_config_type poisson \
  --poisson_request_interval_generator_config_qps 6.45 \
  --replica_scheduler_config_type sarathi \
  --sarathi_scheduler_config_batch_size_cap 512 \
  --sarathi_scheduler_config_chunk_size 512
```

### Format and lint
```bash
make format        # isort + black
make lint          # black + isort checks (what CI runs)
make lint/flake8   # flake8 style checks
```

There is no test suite — validation is empirical via simulation output.

## Architecture

### Simulation loop (`vidur/simulator.py`)

The simulator is **event-driven** with a min-heap priority queue. Each iteration pops the earliest event, advances `current_time`, and calls `event.handle_event()`. Events generate new events, driving the simulation forward.

Event flow for a request:
1. `RequestArrivalEvent` → request added to global scheduler queue
2. `GlobalScheduleEvent` → global scheduler picks a replica, emits `ReplicaScheduleEvent`
3. `ReplicaScheduleEvent` → replica scheduler forms a batch, emits `BatchStageArrivalEvent`
4. `BatchStageArrivalEvent` / `BatchStageEndEvent` → pipeline stage execution
5. `BatchEndEvent` → request completion, metrics recorded

### Key directories

| Path | Purpose |
|------|---------|
| `vidur/entities/` | Core data structures: `Request`, `Batch`, `BatchStage`, `Replica`, `Cluster` |
| `vidur/events/` | Event types — each has a `handle_event()` method |
| `vidur/config/` | Hierarchical dataclass configs flattened to CLI args |
| `vidur/scheduler/global_scheduler/` | Replica selection (random, round-robin, LOR) |
| `vidur/scheduler/replica_scheduler/` | Request batching within a replica (Sarathi, vLLM, Orca, etc.) |
| `vidur/scheduler/replica_stage_scheduler/` | Pipeline stage scheduling |
| `vidur/request_generator/` | Synthetic and trace-replay workload generation |
| `vidur/execution_time_predictor/` | ML models (random forest, linear) predicting GPU execution time |
| `vidur/metrics/` | Metrics collection, CDF sketches, W&B logging, plot generation |
| `vidur/profiling/` | GPU profiling scripts (MLP, attention, collectives, CPU overhead) |
| `vidur/config_optimizer/` | Config space search and analysis |
| `vidur/types/` | Enums for all type selectors (scheduler type, device SKU, etc.) |
| `vidur/utils/base_registry.py` | Registry/factory base class used by all component registries |

### Configuration system (`vidur/config/`)

All config is dataclass-based. `SimulationConfig` aggregates sub-configs (replica, cluster, scheduler, request generator, metrics, etc.). `flat_dataclass.py` flattens nested dataclasses into `--prefixed_arg_name` CLI args. Polymorphic configs use `BasePolyConfig` with type-specific subclasses — the `type` field selects which subclass applies.

### Extending the simulator

All major components use the registry pattern (`BaseRegistry`). To add a new scheduler, predictor, or generator:
1. Create a new class extending the appropriate base (e.g., `BaseReplicaScheduler`)
2. Register it in the corresponding `*_registry.py`
3. Add its type to the relevant enum in `vidur/types/`
4. Add its config dataclass if it has parameters

### Output

Simulation results go to `simulator_output/<TIMESTAMP>/`:
- `event_trace.json` — full event log
- `chrome_trace.json` — view at `chrome://tracing`
- `plots/` — PNG metric visualizations
- `metrics_table/` — CSV tables

Key metrics: TTFT (time to first token), TPOT (time per output token), request E2E time, batch sizes, MFU. See `docs/metrics.md` for definitions.
