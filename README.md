![](banner.jpg)

# News Ranker

A news article ranking system that learns your preferences. Score articles from 1.0 to 10.0, and the system learns to predict scores for new articles based on your feedback.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

1. Start the server:
   ```bash
   ./run serve
   ```

2. Score an article:
   ```bash
   ./run rank "https://example.com/article"
   ```

3. Provide feedback to train the model:
   ```bash
   curl -X POST http://localhost:19091/correct_rank \
     -H "Content-Type: application/json" \
     -d '{"url": "https://example.com/article", "score": 8.5}'
   ```

## Usage

### Server

Start the ranking server (runs on port 19091):

```bash
./run serve
```

### Ranking Articles

Get a score for any article URL:

```bash
./run rank "https://news.ycombinator.com/item?id=12345"
```

Or via the API directly:

```bash
curl "http://localhost:19091/rank?url=https://example.com/article"
```

Response:
```json
{
  "url": "https://example.com/article",
  "rank": 7.5,
  "source": "inference"
}
```

### Training the Model

#### Single Article Feedback

Provide a score (1.0-10.0) for an article via the API:

```bash
curl -X POST http://localhost:19091/correct_rank \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article", "score": 9.0}'
```

#### Bulk Training Import

Import multiple URL/score pairs from a file:

```bash
./run train-bulk -f training_data.txt
```

File format (one entry per line):
```
https://example.com/great-article,9.5
https://example.com/mediocre-article,5.0
https://example.com/bad-article,2.0
```

Or enter data interactively:

```bash
./run train-bulk
```

#### Fetch Training Content

Fetch article text for training URLs that don't have it yet:

```bash
./run fetch-training
```

#### Intensive Retrain

Trigger a full retraining cycle with all available data:

```bash
./run retrain
```

### Statistics

View training data statistics:

```bash
./run stats
```

Output:
```
=== News Ranker Statistics ===

Training samples:   42
With embeddings:    38
Avg target score:   6.75
High quality (8+):  15
Low quality (<=3):  8
```

### Development Commands

```bash
./run lint          # Run code linter
./run test <target> # Run specific test (e.g., src/news_ranker_test.py::test_scoring)
./run check         # Run full test suite and quality gates
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/rank?url=<url>` | GET | Get score for an article URL |
| `/correct_rank` | POST | Submit feedback with `{"url": "...", "score": N}` |
| `/train_bulk` | POST | Import multiple training items |
| `/fetch_training` | POST | Fetch text for training URLs |
| `/retrain` | POST | Trigger intensive retraining |
| `/stats` | GET | Get training data statistics |
| `/shutdown` | POST | Gracefully stop the server |

## Example Workflow

```bash
# Start the server
./run serve

# Score some articles
./run rank "https://news.ycombinator.com/item?id=12345"
./run rank "https://example.com/tech-article"

# The default model gives generic scores. Train it with your preferences:
curl -X POST http://localhost:19091/correct_rank \
  -H "Content-Type: application/json" \
  -d '{"url": "https://news.ycombinator.com/item?id=12345", "score": 9.5}'

# Import more training data from a file
./run train-bulk -f my_ratings.txt

# Fetch content for all training URLs
./run fetch-training

# Run intensive retraining
./run retrain

# Check statistics
./run stats

# Now score new articles - predictions will reflect your preferences
./run rank "https://example.com/new-article"
```