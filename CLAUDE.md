# News Score

ML-powered news article ranking service using sentence transformers and hierarchical attention.

## Quick Start

```bash
./run serve      # Start server on port 19091
./run stats      # Show training statistics
./run rank URL   # Rank a URL
./run check      # Run lint + tests
```

## Architecture

- **Server**: FastAPI with uvicorn on port 19091
- **ML**: sentence-transformers (all-mpnet-base-v2) + custom HierarchicalAttentionRanker
- **Database**: SQLite at `local/news_ranker.db`
- **Worker Pool**: 6 parallel workers with 10-minute rotation for memory management

## Key Files

- `run` - Bash wrapper that activates `.venv` and calls `run.py`
- `run.py` - CLI commands (serve, rank, train, stats, etc.)
- `src/news_ranker.py` - Main server with ML models, worker pool, training loop
- `src/news_ranker_test.py` - Tests (run with `./run check`)

## Virtual Environment

The project requires a `.venv/` with torch, sentence-transformers, playwright, etc. The `run` script automatically uses this venv.

To recreate:
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/playwright install chromium
```

## Service Management (auto)

```bash
~/bin/auto ps                    # Check status
~/bin/auto start news-score      # Start service
~/bin/auto restart news-score    # Restart service
~/bin/auto log news-score        # View logs
```

## Crash Protection

The server has built-in crash protection:
- Catches unhandled exceptions and auto-restarts (up to 10 attempts)
- Exponential backoff: 5s → 10s → 20s → ... → 300s max
- Combined with `auto` daemon manager for double resilience

## Training

- Corrections trigger 50 epochs of incremental training
- `./run retrain` queues 500 epochs of intensive training
- Scheduled retrain runs every 2 hours automatically
- Training runs in background thread, doesn't block ranking

## Gotchas

- SentenceTransformer is NOT thread-safe - each worker has its own embedder instance
- Workers are cycled every 10 minutes to prevent memory accumulation
- The run script must be executed from project root (it changes cwd on startup)

## CPU Usage Patterns

The background threads are designed to minimize CPU usage when idle:

- **Training loop**: Uses blocking `queue.get(timeout=5.0)` when no training queued. Do NOT use spinning `get_nowait()` patterns.
- **Worker pool**: 5-second idle timeout (not 1s) - with 6 workers, shorter timeouts cause excessive wake-ups
- **Epoch throttling**: 10ms sleep between training epochs prevents 100% CPU during training
- **MSELoss**: Instantiated once per training session, not per epoch
