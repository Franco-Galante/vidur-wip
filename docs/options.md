# **Vidur Quick Guide and Needed Extensions**

We discuss the Vidur options, used to configure the simulator, grouping them by function.
Also, we discuss what should be added to match or theoretical model.

## Arrival Process

How the workload enters the system.

- **Request generator** $\rightarrow$ *what* jobs exist and how many (e.g., synthetic, from trace, etc.)
- **Interval generator** $\rightarrow$ *when* jobs arrive (e.g., Poisson)
- **Global stopping**
    - `--synthetic_request_generator_config_num_requests` (if `--request_generator_config_type` is `synthetic`)
    - `--synthetic_request_generator_config_duration`
    - `--time_limit`
- **Seed** golobal seed set with `--seed` 

#### Parameters
- `--request_generator_config_type`
- `--interval_generator_config_type`
    - `--poisson_request_interval_generator_config_qps` (if `--interval_generator_config_type` is `poisson`)
    - `--poisson_request_interval_generator_config_seed`

##### What to extend?

- Nothing so far.


## Job Size (input-output distribution)

This controls the input/outputs lengths.
The simulator generates the lengths in two stages: 1. inter-arrival times (description above) and 2. input-output lengths for each job (here).

*Goal* have a user-defined distribution of (i: *prefill tokens*, o: *decode tokens*) pairs for the jobs in the workload.

#### Supported Modes

- `--fixed_request_length_generator_config_prefill_tokens`
    - `fixed` deterministic pair (i,o)
    - synthetic family, for example `uniform` or `zipf` + **prefill:decode ratio** 
    - `trace` (**Note**: we may synthetically generate the (i,o) pairs from the distribution we prefer and use that as input)


#### `fixed` Mode
/ `--fixed_request_length_generator_config_prefill_tokens` number of input (prompt) tokens
- `--fixed_request_length_generator_config_decode_tokens` number of output (generated) tokens

#### `uniform` Mode

The uniform values are sampled between a *min* and *max* values which are tunable:

- `--uniform_request_length_generator_config_min_tokens` minimum number of tokens
- `--uniform_request_length_generator_config_max_tokens` maximum number of tokens

The link between prompt and generated tokens are defined by:
- `--uniform_request_length_generator_config_prefill_to_decode_ratio` (e.g., smalle decode dominates)

#### `zipf`

**T.B.D.**


#### `trace` Mode

It is possible to provide a CSV file with the (i,o) pairs for each job in the workload.

- `--trace_request_length_generator_config_trace_file` reads a CSV file with paired request lengths using `pandas.read_csv`, then returns row-by-row `num_prefill_tokens` and `num_decode_tokens`.


#### Other `trace` Options

I would keep them as the default parameters at least for the first experiments.

- `--trace_request_length_generator_config_prefill_scale_factor` Vidur multiplies `num_prefill_tokens` for `prefill_scale_factor`
- `--trace_request_length_generator_config_decode_scale_factor` Vidur multiplies `num_decode_tokens` for `decode_scale_factor`


## **Queuing and Scheduling**

This controls how the jobs are scheduled on the system.
*This may be the piece of code I should extend/modify the most since our contribution lays mostly at this level*. 

Vidur scheduling model is split into two levels, which are entry points into richer scheduling layers:

1. **Global Scheduler** which dealt with *cluster-level* routing, i.e., which **replica** (model instance + GPU workers, **def.** *a replica is a deployed copy of the model (often tied to a GPU or a group of GPUs)*) should a job be routed to. [**Note** in the early stages we may consider a single replica on a single GPU to reduce complexity]

    The *goal* of the global scheduler is: i) load across the replicas (e.g., avoid overloaded replicas), ii) consider possible affinity (e.g., caching, KV cache reutilization, ...)

2. **Replica Scheduler** this is the scheduler that manages the scheduling of the jobs when they **land a particular *replica***. 
    It decides: 
    - **Queuing**, FIFO? priority queue? separate queues for prefill and decode?
    - **Admission control**, when the job gets served? Do we delay the service to *batch* it with others?
    - **Batching**: which requests get grouped together? *dynamic batching* is critical
    - *General*, how prefill nad decode phases are interleaved?

| Layer             | Scope              | Key question                              |
| ----------------- | ------------------ | ----------------------------------------- |
| Global scheduler  | Across replicas    | “Where should this request go?”           |
| Replica scheduler | Inside one replica | “How do I execute all assigned requests?” |

#### Parameters

- `--global_scheduler_config_type` 
    - RANDOM
    - ROUND_ROBIN
    - LOR
    Setting `num_replicas=1` renders this level inert

- `--replica_scheduler_config_type`, control the local scheduling of the replica
    - `FASTER_TRANSFORMER`
    - `ORCA`
    - `SARATHI`
    - `VLLM`
    - `LIGHTLLM`

    These are not just generic queuing disciplines, there are **engine-inspired scheduling models**. This *local scheduler* is actually responsible for:
    - queue order
    - admission *s.t. memory*
    - *count* of allocated blocks
    - batch formation
    - when to free memory (preemption?)

    **Note** Vidur is designed to be **extensible** so it is possible to add new scheduling algorithms here. How to:
    - a new scheduler implementation under `vidur/scheduler/replica_scheduler/`
    - a new enum entry in `ReplicaSchedulerType`
    - the corresponding registry/config wiring so the simulator can instantiate it from config/CLI. The codebase already has a `BaseReplicaScheduler` abstraction and a dedicated replica-scheduler directory, which is the extension point.

    [Local admission knobs] All replica scheduler inherit a base configuration (controlling the *admission*):
    - `batch_size_cap`, upper bound on how many sequences can be put into a batch
    - `block_size`, granularity of memory/KV allocation
    - `watermark_blocks_fraction`, a memory-reservation / safety-margin style parameter available in the base replica scheduler config (**??**)
    - `num_blocks`, total allocatable blocks in the replica
