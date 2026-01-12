import os
import re
import sqlite3
import pickle
import asyncio
import signal
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, Query
from pydantic import BaseModel
import uvicorn
import setproctitle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sentence_transformers import SentenceTransformer
import trafilatura
from playwright.sync_api import sync_playwright


# ##################################################################
# path resolution
# compute paths relative to this source file so db location is consistent
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
LOCAL_DIR = PROJECT_ROOT / "local"


# ##################################################################
# ensure local directory
# creates the local directory if it does not exist
def ensure_local_directory() -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)


# ##################################################################
# configuration
# device detection and constants for the ranking system
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DB_FILE = str(LOCAL_DIR / "news_ranker.db")
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
EMBEDDING_DIM = 768
EMBEDDING_MODEL_VERSION = 2
CHUNK_SIZE_WORDS = 500
MAX_SEQ_LENGTH = 512
TRAINING_EPOCHS = 1000
INTENSIVE_TRAINING_EPOCHS = 40000
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.1
HIDDEN_DIM = 128
MIN_SCORE = 1.0
MAX_SCORE = 10.0
DEFAULT_SCORE = 5.5
SERVER_PORT = 19091


# ##################################################################
# initialize database
# creates the articles table if it does not exist
def initialize_database() -> None:
    ensure_local_directory()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            url TEXT PRIMARY KEY,
            extracted_text TEXT,
            embedding BLOB,
            current_rank REAL,
            user_target_rank REAL,
            is_training_data INTEGER DEFAULT 0
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS models (
            name TEXT PRIMARY KEY,
            weights BLOB,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    conn.commit()
    conn.close()
    check_model_version_migration()


# ##################################################################
# get database connection
# returns a new sqlite connection for thread safety
def get_database_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_FILE)


# ##################################################################
# check model version migration
# clears embeddings and trained model if embedding model version changed
def check_model_version_migration() -> None:
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = 'embedding_model_version'")
    row = cursor.fetchone()
    stored_version = int(row[0]) if row else 0

    if stored_version != EMBEDDING_MODEL_VERSION:
        print(f"Embedding model version changed ({stored_version} -> {EMBEDDING_MODEL_VERSION}). Clearing old data...")
        cursor.execute("UPDATE articles SET embedding = NULL")
        cursor.execute("DELETE FROM models WHERE name = 'ranker'")
        cursor.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES ('embedding_model_version', ?)",
            (str(EMBEDDING_MODEL_VERSION),)
        )
        conn.commit()
        print("Old embeddings and trained model cleared. Run fetch-training and retrain.")
    conn.close()


# ##################################################################
# save model weights
# serializes model state dict to database
def save_model_weights(name: str, state_dict: dict) -> None:
    conn = get_database_connection()
    cursor = conn.cursor()
    weights_blob = pickle.dumps({k: v.cpu() for k, v in state_dict.items()})
    cursor.execute(
        "INSERT OR REPLACE INTO models (name, weights, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
        (name, weights_blob)
    )
    conn.commit()
    conn.close()
    print(f"Model '{name}' saved to database.")


# ##################################################################
# load model weights
# deserializes model state dict from database
def load_model_weights(name: str) -> Optional[dict]:
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT weights FROM models WHERE name = ?", (name,))
    row = cursor.fetchone()
    conn.close()
    if row and row[0]:
        return pickle.loads(row[0])
    return None


# ##################################################################
# ranking head
# neural network that converts embeddings to a scalar score
class RankingHead(nn.Module):

    # ##################################################################
    # init
    # sets up layers with bias initialized to output default score
    def __init__(self, input_dim: int) -> None:
        super(RankingHead, self).__init__()
        self.hidden = nn.Linear(input_dim, HIDDEN_DIM)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.output = nn.Linear(HIDDEN_DIM, 1)
        nn.init.zeros_(self.hidden.weight)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, DEFAULT_SCORE)

    # ##################################################################
    # forward
    # runs input through the network layers
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.output(x)


# ##################################################################
# scoring engine
# handles embedding, inference, and training for article ranking
class ScoringEngine:

    # ##################################################################
    # init
    # loads the sentence transformer and ranking head onto the device
    def __init__(self) -> None:
        print(f"Loading models to {DEVICE}...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
        self.embedder.max_seq_length = MAX_SEQ_LENGTH
        self.ranker = RankingHead(EMBEDDING_DIM).to(DEVICE)
        saved_weights = load_model_weights("ranker")
        if saved_weights:
            self.ranker.load_state_dict(saved_weights)
            print("Loaded trained model weights from database.")
        else:
            print("No saved weights found, using fresh model.")
        self.ranker.eval()
        self.is_training = False

    # ##################################################################
    # chunk text
    # splits text into word chunks for processing long articles
    def chunk_text(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return []
        return [" ".join(words[i:i + CHUNK_SIZE_WORDS]) for i in range(0, len(words), CHUNK_SIZE_WORDS)]

    # ##################################################################
    # embed chunks
    # encodes text chunks into tensor embeddings
    def embed_chunks(self, chunks: list[str]) -> torch.Tensor:
        return self.embedder.encode(chunks, convert_to_tensor=True, show_progress_bar=False)

    # ##################################################################
    # average embeddings
    # computes mean of chunk embeddings for final article vector
    def average_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.dim() > 1 and embeddings.shape[0] > 1:
            return torch.mean(embeddings, dim=0)
        return embeddings.squeeze(0) if embeddings.dim() > 1 else embeddings

    # ##################################################################
    # vectorize text
    # converts text to a single embedding vector via chunking and averaging
    def vectorize_text(self, text: str) -> torch.Tensor:
        chunks = self.chunk_text(text)
        if not chunks:
            return torch.zeros(EMBEDDING_DIM, device=DEVICE)
        embeddings = self.embed_chunks(chunks)
        return self.average_embeddings(embeddings)

    # ##################################################################
    # clamp score
    # ensures score falls within valid range
    def clamp_score(self, score: float) -> float:
        return max(MIN_SCORE, min(MAX_SCORE, score))

    # ##################################################################
    # predict score
    # runs inference to get a clamped score for text
    def predict_score(self, text: str) -> float:
        with torch.no_grad():
            vector = self.vectorize_text(text)
            vector_batch = vector.unsqueeze(0)
            raw_score = self.ranker(vector_batch).item()
            return self.clamp_score(raw_score)

    # ##################################################################
    # get embedding bytes
    # returns pickled embedding for database storage
    def get_embedding_bytes(self, text: str) -> bytes:
        with torch.no_grad():
            vector = self.vectorize_text(text)
            return pickle.dumps(vector.cpu())

    # ##################################################################
    # load training data
    # fetches all articles with user-provided target ranks from database
    def load_training_data(self) -> list[tuple]:
        conn = get_database_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT embedding, user_target_rank FROM articles WHERE user_target_rank IS NOT NULL AND embedding IS NOT NULL"
        )
        rows = cursor.fetchall()
        conn.close()
        return rows

    # ##################################################################
    # prepare training tensors
    # deserializes embeddings and creates training tensors
    def prepare_training_tensors(self, rows: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
        embedding_list = [pickle.loads(row[0]) for row in rows]
        target_list = [row[1] for row in rows]
        x_train = torch.stack(embedding_list).to(DEVICE)
        y_train = torch.tensor(target_list, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        return x_train, y_train

    # ##################################################################
    # create new ranker
    # creates a copy of the current ranker for training
    def create_new_ranker(self) -> RankingHead:
        new_ranker = RankingHead(EMBEDDING_DIM).to(DEVICE)
        new_ranker.load_state_dict(self.ranker.state_dict())
        new_ranker.train()
        return new_ranker

    # ##################################################################
    # train ranker
    # runs training loop on the new ranker
    def train_ranker(self, ranker: RankingHead, x_train: torch.Tensor, y_train: torch.Tensor) -> float:
        criterion = nn.MSELoss()
        optimizer = optim.Adam(ranker.parameters(), lr=LEARNING_RATE)
        scheduler = CosineAnnealingLR(optimizer, T_max=TRAINING_EPOCHS)
        loss = None
        for _ in range(TRAINING_EPOCHS):
            optimizer.zero_grad()
            outputs = ranker(x_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            scheduler.step()
        return loss.item() if loss else 0.0

    # ##################################################################
    # retrain
    # trains a new model on user feedback and hot-swaps it in
    def retrain(self) -> None:
        if self.is_training:
            print("Training already in progress. Skipping.")
            return

        self.is_training = True
        try:
            rows = self.load_training_data()
            if not rows:
                print("No training data found.")
                return

            print(f"Retraining on {len(rows)} samples...")
            x_train, y_train = self.prepare_training_tensors(rows)
            new_ranker = self.create_new_ranker()
            final_loss = self.train_ranker(new_ranker, x_train, y_train)

            new_ranker.eval()
            self.ranker = new_ranker
            save_model_weights("ranker", self.ranker.state_dict())
            print(f"Training complete. Final Loss: {final_loss:.4f}")
        finally:
            self.is_training = False

    # ##################################################################
    # generate missing embeddings
    # creates embeddings for training data that lacks them
    def generate_missing_embeddings(self) -> int:
        conn = get_database_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT url, extracted_text FROM articles WHERE user_target_rank IS NOT NULL AND embedding IS NULL"
        )
        rows = cursor.fetchall()

        count = 0
        for url, text in rows:
            if not text:
                print(f"  Skipping {url[:50]}... (no text)")
                continue
            embedding_blob = self.get_embedding_bytes(text)
            cursor.execute("UPDATE articles SET embedding = ? WHERE url = ?", (embedding_blob, url))
            count += 1
            print(f"  Generated embedding for {url[:50]}...")

        conn.commit()
        conn.close()
        return count

    # ##################################################################
    # intensive retrain
    # resets model and trains from scratch with more epochs
    def intensive_retrain(self) -> None:
        if self.is_training:
            print("Training already in progress. Skipping.")
            return

        self.is_training = True
        try:
            missing = self.generate_missing_embeddings()
            if missing > 0:
                print(f"Generated {missing} missing embeddings")

            rows = self.load_training_data()
            rows = [(emb, target) for emb, target in rows if emb is not None]

            if not rows:
                print("No training data with embeddings found.")
                return

            print(f"Intensive retraining on {len(rows)} samples...")
            print(f"Using {INTENSIVE_TRAINING_EPOCHS} epochs...")
            x_train, y_train = self.prepare_training_tensors(rows)

            new_ranker = self.create_new_ranker()

            criterion = nn.MSELoss()
            optimizer = optim.Adam(new_ranker.parameters(), lr=LEARNING_RATE)
            scheduler = CosineAnnealingLR(optimizer, T_max=INTENSIVE_TRAINING_EPOCHS)

            best_loss = float("inf")
            best_state = None
            patience = 5000
            epochs_without_improvement = 0

            for epoch in range(INTENSIVE_TRAINING_EPOCHS):
                optimizer.zero_grad()
                outputs = new_ranker(x_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
                scheduler.step()

                current_loss = loss.item()
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_state = {k: v.clone() for k, v in new_ranker.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if (epoch + 1) % 1000 == 0:
                    lr = scheduler.get_last_lr()[0]
                    print(f"  Epoch {epoch + 1}/{INTENSIVE_TRAINING_EPOCHS}, Loss: {current_loss:.4f}, Best: {best_loss:.4f}, LR: {lr:.6f}")

                if epochs_without_improvement >= patience:
                    print(f"  Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
                    break

            if best_state:
                new_ranker.load_state_dict(best_state)
            new_ranker.eval()
            self.ranker = new_ranker
            save_model_weights("ranker", self.ranker.state_dict())
            print(f"Intensive training complete. Best Loss: {best_loss:.4f}")
        finally:
            self.is_training = False


# ##################################################################
# set training target
# sets user target rank for a url in the database
def set_training_target(url: str, score: float) -> bool:
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT url FROM articles WHERE url = ?", (url,))
    exists = cursor.fetchone()

    if exists:
        cursor.execute("UPDATE articles SET user_target_rank = ? WHERE url = ?", (score, url))
    else:
        cursor.execute(
            "INSERT INTO articles (url, user_target_rank) VALUES (?, ?)",
            (url, score)
        )
    conn.commit()
    conn.close()
    return True


# ##################################################################
# get training stats
# returns count and summary of training data
def get_training_stats() -> dict:
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM articles")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM articles WHERE user_target_rank IS NOT NULL")
    trained = cursor.fetchone()[0]
    cursor.execute("SELECT AVG(user_target_rank) FROM articles WHERE user_target_rank IS NOT NULL")
    avg = cursor.fetchone()[0]
    conn.close()
    return {"total_articles": total, "trained_articles": trained, "avg_target": avg}


# global engine instance
engine: Optional[ScoringEngine] = None


# ##################################################################
RETRAIN_INTERVAL_SECONDS = 2 * 60 * 60


# ##################################################################
# scheduled retrain task
# runs retrain every 2 hours in the background
async def scheduled_retrain_task() -> None:
    while True:
        await asyncio.sleep(RETRAIN_INTERVAL_SECONDS)
        print("Scheduled retrain starting...")
        try:
            engine.retrain()
        except Exception as e:
            print(f"Scheduled retrain failed: {e}")


# ##################################################################
# lifespan
# manages startup and shutdown of the fastapi application
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    initialize_database()
    engine = ScoringEngine()
    retrain_task = asyncio.create_task(scheduled_retrain_task())
    print("System initialized and ready. Scheduled retrain every 2 hours.")
    yield
    retrain_task.cancel()
    print("Shutting down system...")


app = FastAPI(title="NewsRanker", lifespan=lifespan)


# ##################################################################
# correct rank request
# pydantic model for training feedback endpoint
class CorrectRankRequest(BaseModel):
    url: str
    score: float


# ##################################################################
# rank response
# pydantic model for rank endpoint responses
class RankResponse(BaseModel):
    url: str
    rank: float
    source: str


# ##################################################################
# fetch with trafilatura
# fast extraction for static html pages
def fetch_with_trafilatura(url: str) -> Optional[str]:
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return None
        extracted = trafilatura.extract(downloaded)
        return extracted if extracted else None
    except Exception:
        return None


# ##################################################################
# clean html text
# removes extra whitespace and normalizes text extracted from html
def clean_html_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ##################################################################
# fetch with playwright
# uses headless browser for javascript-rendered pages
def fetch_with_playwright(url: str) -> Optional[str]:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            )
            page = context.new_page()
            page.goto(url, timeout=30000, wait_until="networkidle")
            page.wait_for_timeout(2000)
            content = page.content()
            browser.close()

        extracted = trafilatura.extract(content)
        if extracted:
            return extracted

        text = clean_html_text(content)
        return text if len(text) > 100 else None
    except Exception:
        return None


# ##################################################################
# fetch article text
# tries trafilatura first, falls back to playwright for js-rendered pages
async def fetch_article_text(url: str) -> Optional[str]:
    loop = asyncio.get_event_loop()

    text = await loop.run_in_executor(None, fetch_with_trafilatura, url)
    if text and len(text) > 100:
        return text

    text = await loop.run_in_executor(None, fetch_with_playwright, url)
    return text


# ##################################################################
# rank article endpoint
# fetches and scores url fresh every time, always returns 200
@app.get("/rank", response_model=RankResponse)
async def rank_article(url: str = Query(..., min_length=1)):
    try:
        extracted_text = await fetch_article_text(url)
        if not extracted_text:
            return {"url": url, "rank": 0.0, "source": "unavailable"}

        score = engine.predict_score(extracted_text)
        return {"url": url, "rank": score, "source": "ok"}
    except Exception:
        return {"url": url, "rank": 0.0, "source": "error"}


# ##################################################################
# check article exists
# returns extracted text if article is in database, none otherwise
def check_article_exists(url: str) -> Optional[str]:
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT extracted_text FROM articles WHERE url = ?", (url,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


# ##################################################################
# store new article with target
# inserts a new article with user feedback
def store_new_article_with_target(url: str, text: str, embedding_blob: bytes, target_score: float) -> None:
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO articles (url, extracted_text, embedding, user_target_rank)
        VALUES (?, ?, ?, ?)
    ''', (url, text, embedding_blob, target_score))
    conn.commit()
    conn.close()


# ##################################################################
# update article target
# updates the user target rank for an existing article
def update_article_target(url: str, target_score: float) -> None:
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute('UPDATE articles SET user_target_rank = ? WHERE url = ?', (target_score, url))
    conn.commit()
    conn.close()


# ##################################################################
# correct rank endpoint
# accepts user feedback and triggers retraining, always returns 200
@app.post("/correct_rank")
async def correct_rank(payload: CorrectRankRequest, background_tasks: BackgroundTasks):
    try:
        existing_text = check_article_exists(payload.url)

        if not existing_text:
            text = await fetch_article_text(payload.url)
            if not text:
                return {"status": "skipped", "message": "Could not extract text from URL."}
            embedding_blob = engine.get_embedding_bytes(text)
            store_new_article_with_target(payload.url, text, embedding_blob, payload.score)
        else:
            update_article_target(payload.url, payload.score)

        background_tasks.add_task(engine.retrain)
        return {"status": "accepted", "message": "Feedback received. Retraining initiated."}
    except Exception:
        return {"status": "error", "message": "Could not process request."}


# ##################################################################
# shutdown endpoint
# gracefully terminates the server process
@app.post("/shutdown")
async def shutdown():
    os.kill(os.getpid(), signal.SIGTERM)
    return {"message": "Server shutting down..."}


# ##################################################################
# train bulk request
# pydantic model for bulk training endpoint
class TrainBulkRequest(BaseModel):
    items: list[dict]


# ##################################################################
# train bulk endpoint
# accepts list of url/score pairs for bulk training
@app.post("/train_bulk")
async def train_bulk(payload: TrainBulkRequest, background_tasks: BackgroundTasks):
    try:
        count = 0
        for item in payload.items:
            url = item.get("url")
            score = item.get("score")
            if not url or score is None:
                continue
            set_training_target(url, float(score))
            count += 1

        if count > 0:
            background_tasks.add_task(engine.retrain)

        return {"status": "accepted", "count": count, "message": f"Added {count} training targets."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ##################################################################
# fetch training endpoint
# fetches text for all training urls that lack it
@app.post("/fetch_training")
async def fetch_training_endpoint():
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT url FROM articles WHERE user_target_rank IS NOT NULL AND extracted_text IS NULL"
        )
        urls = [row[0] for row in cursor.fetchall()]
        conn.close()

        if not urls:
            return {"status": "ok", "fetched": 0, "message": "All training URLs already have text."}

        fetched = 0
        failed = 0
        for url in urls:
            print(f"Fetching: {url[:60]}...")
            text = await fetch_article_text(url)
            if text:
                embedding_blob = engine.get_embedding_bytes(text)
                conn = get_database_connection()
                conn.execute(
                    "UPDATE articles SET extracted_text = ?, embedding = ? WHERE url = ?",
                    (text, embedding_blob, url)
                )
                conn.commit()
                conn.close()
                fetched += 1
                print(f"  OK ({len(text)} chars)")
            else:
                failed += 1
                print("  FAILED")

        return {
            "status": "ok",
            "fetched": fetched,
            "failed": failed,
            "message": f"Fetched {fetched} URLs, {failed} failed."
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ##################################################################
# retrain endpoint
# triggers intensive retraining from all training data
@app.post("/retrain")
async def retrain_endpoint(background_tasks: BackgroundTasks):
    try:
        stats = get_training_stats()
        if stats["trained_articles"] == 0:
            return {"status": "error", "message": "No training data available."}

        background_tasks.add_task(engine.intensive_retrain)
        return {
            "status": "accepted",
            "training_samples": stats["trained_articles"],
            "message": "Intensive retraining initiated."
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ##################################################################
# stats endpoint
# returns training statistics
@app.get("/stats")
async def stats_endpoint():
    try:
        conn = get_database_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM articles WHERE user_target_rank IS NOT NULL")
        trained = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM articles WHERE user_target_rank IS NOT NULL AND embedding IS NOT NULL")
        with_embeddings = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(user_target_rank) FROM articles WHERE user_target_rank IS NOT NULL")
        avg_target = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM articles WHERE user_target_rank >= 8")
        high_quality = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM articles WHERE user_target_rank <= 3")
        low_quality = cursor.fetchone()[0]

        conn.close()

        return {
            "training_samples": trained,
            "with_embeddings": with_embeddings,
            "avg_target_score": round(avg_target, 2) if avg_target else None,
            "high_quality_count": high_quality,
            "low_quality_count": low_quality
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ##################################################################
# main
# entry point that starts the uvicorn server
def main() -> None:
    setproctitle.setproctitle("news-ranker-server")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, workers=1)


# ##################################################################
# entry point
# standard python pattern for running as script
if __name__ == "__main__":
    main()
