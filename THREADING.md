# Threading Model

## Overview

The news-score system uses a **worker pool architecture** for true parallel ranking:
- 6 independent ranking workers, each with its own SentenceTransformer
- Workers pull from a shared request queue - true parallelism, no lock contention
- Single training thread processes training requests from a queue (fire-and-forget)
- Workers cycle out gracefully every 10 minutes to stay fresh

## Architecture

```
                    ┌─────────────────┐
                    │   /rank API     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Request Queue  │
                    └────────┬────────┘
          ┌──────────┬───────┼───────┬──────────┐
          │          │       │       │          │
     ┌────▼───┐ ┌────▼───┐ ┌─▼──┐ ┌──▼───┐ ┌────▼───┐
     │Worker 0│ │Worker 1│ │ .. │ │ .. 4 │ │Worker 5│
     │        │ │        │ │    │ │      │ │        │
     │Embedder│ │Embedder│ │    │ │      │ │Embedder│
     └────────┘ └────────┘ └────┘ └──────┘ └────────┘
          │          │       │       │          │
          └──────────┴───────┼───────┴──────────┘
                             │
                    ┌────────▼────────┐
                    │ Response Events │
                    └─────────────────┘
```

## Worker Pool

Each worker has:
- **Own SentenceTransformer** - no lock contention for embeddings
- **Shared state snapshot** - ranker model, calibration, cached doc vectors (read-only)
- **10-minute lifespan** - cycles out gracefully to pick up training updates

### Worker Lifecycle

1. Worker starts, loads its own embedder
2. Pulls requests from queue, processes them independently
3. After 10 minutes, receives graceful shutdown signal
4. Finishes current task, then exits
5. Next cycle check spawns replacement

### Graceful Cycling

Every minute, the cycling task:
1. Counts alive workers
2. If at target (6): requests shutdown on oldest expired worker
3. If below target: spawns new workers

Workers **never** shut down mid-task. They finish their current ranking, then exit on the next loop iteration.

## Request Flow

```python
# 1. Request comes in
request_id = worker_pool.submit_request(text)

# 2. Any available worker picks it up
request = self.request_queue.get()
score = worker.predict_score(request.text)

# 3. Response goes to event map
self.response_events[request_id].set()

# 4. Caller gets notified
response = worker_pool.get_response(request_id, timeout=30)
```

## Training Queue (Fire-and-Forget)

Training requests are queued, not executed inline:

```python
# Caller doesn't wait
training_queue.submit_quick_retrain()  # Returns immediately
training_queue.submit_intensive_retrain()  # Returns immediately

# Single background thread processes queue
def _training_loop(self):
    while self.running:
        training_type = self.queue.get()
        if training_type == "intensive":
            self.engine.intensive_retrain()
        else:
            self.engine.retrain()
        # Refresh worker shared state after training
        self.worker_pool.refresh_shared_state()
```

## Shared State

Workers share read-only state (updated atomically after training):

```python
shared_state = {
    "ranker": engine.ranker,           # Model for inference
    "calibration_scale": 1.0,          # Score calibration
    "calibration_bias": 0.0,
    "user_profile": cached_tensor,     # User preference vector
    "doc_vectors": cached_list,        # Training doc vectors for similarity
}
```

After training completes, `refresh_shared_state()` atomically updates the dict reference. Workers snapshot this at prediction start - no locks needed.

## Concurrency Matrix

| Operation | Concurrency |
|-----------|-------------|
| Multiple `/rank` requests | Fully parallel (6 workers, each with own embedder) |
| `/rank` during training | Parallel (workers use snapshot state) |
| Training requests | Queued, single thread processes them |
| Worker cycling | One worker cycles out per minute |

## No Locks for Ranking

The key insight: each worker has its **own** SentenceTransformer instance. No locks are needed because:
1. Workers don't share the embedder
2. Shared state is read-only, updated atomically via dict reference swap
3. Workers snapshot shared state at prediction start

## Timeouts

| Operation | Timeout |
|-----------|---------|
| Ranking request | 30 seconds |
| Trafilatura fetch | 15 seconds |
| Playwright fetch | 45 seconds |
| Worker shutdown | Graceful (no timeout) |

## Configuration

```python
WORKER_POOL_SIZE = 6                    # Number of parallel workers
WORKER_MAX_AGE_SECONDS = 600            # 10 minutes before cycling
WORKER_CYCLE_INTERVAL_SECONDS = 60      # Check every minute
RANKING_TIMEOUT_SECONDS = 30            # Max wait for response
```
