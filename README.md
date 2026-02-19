# weir

A lightweight, opinionated framework for composable async data pipelines in Python.

## Philosophy

**weir** exists in the gap between "just write some asyncio" and "deploy Airflow."

Most data pipelines start as a script. Then they grow. You add retries. Then backpressure
(or more likely, you *don't*, and your script eats all available memory). Then you scatter
try/except blocks everywhere. Then you add logging, and it's inconsistent. Then someone asks
"how many items per second are we processing?" and you realize you have no idea.

**weir** solves this by making a single assertion: **a pipeline is a sequence of
async stages connected by bounded channels.** Everything else — retries, backpressure,
observability, error routing, graceful shutdown — falls out of taking that assertion seriously.

### Design Principles

1. **Composition over configuration.** Stages are async functions. Pipelines are stages
   wired together. No YAML. No DAG definitions. Just Python.

2. **Backpressure is not optional.** Every inter-stage channel is bounded. If a downstream
   stage is slow, upstream stages block. This is a feature. Unbounded queues are a lie —
   they convert memory pressure problems into OOM kills.

3. **Errors are data, not exceptions.** Transient failures retry. Permanent failures route
   to dead letters. Poison pills exhaust retries and route to dead letters. You declare this,
   not code it.

4. **Observability is structural.** Every stage is an instrumentation boundary. Latency,
   throughput, error rates, and queue depth are emitted automatically because the framework
   *is* the measurement point.

5. **Graceful shutdown is a first-class concern.** SIGINT/SIGTERM trigger a drain: stop
   accepting new items, flush in-flight work, flush partial batches, then exit. No data loss.
   No deadlocks.

6. **Concurrency is declared, not discovered.** Each stage declares its parallelism.
   "Hit this API with 5 concurrent requests." "Write to the DB with 2 connections."
   This is configuration, not implementation.

## Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │                  Pipeline                       │
                    │                                                 │
 ┌─────────┐   ┌───▼───┐   ┌─────────┐   ┌───────┐   ┌──────────┐  │
 │  Source  │──▶│Channel│──▶│ Stage A │──▶│Channel│──▶│ Stage B  │  │
 │(iterable)│   │(bound)│   │ (n=5)   │   │(bound)│   │ (n=2)    │  │
 └─────────┘   └───────┘   └────┬────┘   └───────┘   └────┬─────┘  │
                                 │                          │        │
                                 ▼                          ▼        │
                           ┌──────────┐              ┌──────────┐   │
                           │  Metrics │              │  Metrics │   │
                           └──────────┘              └──────────┘   │
                    │                                                 │
                    │   on_error(InvalidData) ──▶ DeadLetterCollector │
                    └─────────────────────────────────────────────────┘
```

### Core Abstractions

- **`Stage`**: A decorated async function with concurrency, retry, and timeout policy.
  The atomic unit of work. Configuration inherits from `RetryConfig`.
- **`BatchStage`**: A decorated async function that processes items in groups. Accumulates
  items until `batch_size` is reached or `flush_timeout` expires, then flushes.
- **`BaseStageRunner`**: Abstract base class providing shared lifecycle management
  (`start`, `wait`, `cancel`) for both `StageRunner` and `BatchStageRunner`.
- **`Channel`**: A bounded async queue connecting two stages. The backpressure mechanism.
  Configurable capacity. Use `is_stop_signal()` to check for shutdown sentinels.
- **`Pipeline`**: A composed sequence of stages and channels. Owns the lifecycle:
  startup, run, drain, shutdown. Hooks are typed via the `Hook` protocol.
- **`StageMetrics`**: Per-stage counters and histograms. Emitted automatically.
- **`ErrorRouter`**: Maps exception types to handlers (retry, dead-letter).
- **`RetryPolicy`**: Exponential backoff with ±50% jitter to prevent thundering herd.

### Data Flow

1. Source emits items into the first channel.
2. Each stage pulls from its input channel, processes (with concurrency semaphore),
   and pushes to its output channel.
3. If the output channel is full, the stage blocks (backpressure).
4. If processing raises, the error router decides: retry? dead-letter?
5. Metrics are recorded at each stage boundary.
6. On shutdown signal, sources stop emitting. Stages drain their input channels.
   Pipeline waits for all in-flight work, then exits cleanly.

## Quick Start

```python
from weir import Pipeline, stage

@stage(concurrency=5, retries=3, timeout=30)
async def fetch(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        return (await client.get(url)).json()

@stage(concurrency=10)
async def transform(data: dict) -> Record:
    return Record.from_raw(data)

@stage(concurrency=2)
async def save(record: Record) -> None:
    await db.insert(record)

pipe = (
    Pipeline("my-ingest")
    .source(url_stream())
    .then(fetch)
    .then(transform)
    .then(save)
    .build()
)

await pipe.run()
```

## Batch Stages

For operations that are more efficient in groups (bulk DB inserts, batch API calls):

```python
from weir import Pipeline, stage, batch_stage

@stage(concurrency=5)
async def fetch(url: str) -> dict:
    ...

@batch_stage(batch_size=50, flush_timeout=2.0, concurrency=2)
async def bulk_insert(items: list[dict]) -> None:
    await db.insert_many(items)

pipe = (
    Pipeline("batch-ingest")
    .source(urls)
    .then(fetch)
    .then(bulk_insert)
    .build()
)

await pipe.run()
```

The `BatchStageRunner` accumulates items until `batch_size` is reached or `flush_timeout`
expires, whichever comes first. On shutdown, any remaining items are flushed as a partial batch.

## Installation

```bash
# From git
pip install git+https://github.com/paulkasay/weir.git

# Local development
pip install -e ".[dev]"
```

## Project Structure

```
src/weir/
├── __init__.py          # Public API surface
├── stage.py             # @stage decorator and StageRunner
├── batch.py             # @batch_stage decorator and BatchStageRunner
├── runner.py            # BaseStageRunner ABC (shared lifecycle logic)
├── channel.py           # Bounded async channel with backpressure
├── pipeline.py          # Pipeline builder and runtime
├── metrics.py           # Per-stage metrics collection
├── errors.py            # Error routing, retry logic, and RetryConfig base class
├── shutdown.py          # Graceful shutdown coordination
├── hooks.py             # Hook protocol for lifecycle extensibility
└── logging.py           # Structured logging setup
```

## License

MIT
