# Eudoxia + Bauplan — Context for Policy Generation

## TL;DR

* Bauplan is a function-first, data-aware FaaS lakehouse: every workload step is a stateless function `Table(s) -> Table`, executed in ephemeral containers with Arrow-native data passing.
* Because everything reduces to running functions, the platform's central challenge is scheduling: deciding which function runs where/when and with how many resources across mixed-priority workloads.
* Eudoxia is a deterministic simulator that models Bauplan's execution so we can test scheduling policies offline—fast, cheap, and safely—before rolling them out.

## What Bauplan Is (what we're scheduling)

* Programming model: pipelines are DAGs of pure functions (SQL/Python) with explicit table inputs/outputs.
* Runtime: functions run as short-lived containers on customer VMs; the control plane plans, workers execute; bi-directional streaming gives an interactive “feels local” dev loop.
* Unification: the same runtime handles interactive queries, iterative development, and batch pipelines; the only knob is scheduling functions.

## Why Scheduling Is Hard (and essential)

* Heterogeneous operators: each function has a minimum RAM requirement (below which it OOMs) and a nonlinear CPU scaling curve (more vCPU does not always give proportional speedups).
* Mixed priorities: interactive, iterative, and batch arrive concurrently; we must protect tail latency for high-priority work without starving low-priority jobs.
* Atomic units: a function is the scheduling atom; it cannot be split across VMs, so placement, admission, and (when needed) preemption are pivotal.
* Scale-up bias: many analytics steps prefer a single beefy VM over scattering; right-sizing containers and pools matters more than micro-sharding.
* Real-time uncertainty: the scheduler rarely knows a function's true resource curve; it must infer from signals (failures, OOMs, queueing, observed runtimes).

## Objectives

* Bound p95/p99 latency for high-priority work.
* Maximize throughput and utilization without cascading OOMs.
* Minimize preemption churn and associated wasted work.
* Respect fairness and avoid indefinite starvation.

## Eudoxia: What It Simulates

* Workload generator: creates pipelines (DAGs) with per-operator RAM minima and CPU scaling; supports random or trace-driven arrivals and priority mixes.
* Scheduler interface: plug-in policies via simple init/step callbacks. The policy decides admission, placement (pool/VM), CPU/RAM sizing, and preemptions.
* Executor model: one or more VM pools with finite vCPU and RAM. Execution time depends on allocated CPU and operator's scaling curve; allocations below RAM minima fail fast as OOM.
* Clock & determinism: a fast, discrete event loop advances the system, collecting comparable metrics for each policy.

## Policies You Can Evaluate (examples)

* Naive FIFO: single pool; admit next job, saturate available resources. Baseline for contention.
* Priority aware (single pool): reserve small initial shares; on OOM, requeue with larger RAM; preempt low-priority when high-priority arrives and needs headroom.
* Priority + multi-pool: choose the pool with most headroom or dedicated “interactive” pool; same OOM and preemption logic.
* Custom variants: SRPT-like CPU awareness, RAM-first packing, priority aging, SLO-aware admission, queue capping, two-level scheduling, etc.

## Metrics Tracked

* Latency: mean, p95, p99 per priority class.
* Throughput & makespan: completed ops/pipelines per unit time; end-to-end completion.

## How to Use This Context
When proposing a novel policy to be tested on Eudoxia, reason about:

* Admission: which pipeline/function enters now, and at what size?
* Placement: which pool/VM minimizes interference and OOM risk?
* Resizing: how to react to OOM/slowdown (RAM-first, CPU-first, or both)?
* Preemption: when to evict low-priority work to protect interactive SLOs?
* Fairness: ensure progress for background/batch workloads.

If provided, use the results of previous simulations to reason about trade-offs, expected outcomes, and plan accordingly novel policy designs to iteratively test and refine the FaaS scheduling.