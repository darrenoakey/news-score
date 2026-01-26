import os
import re
import sqlite3
import pickle
import asyncio
import signal
import threading
import queue
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query
from pydantic import BaseModel
import uvicorn
import setproctitle

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import trafilatura
from playwright.sync_api import sync_playwright


# ##################################################################
# path resolution
# compute paths relative to this source file so db location is consistent
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
LOCAL_DIR = PROJECT_ROOT / "local"
OUTPUT_DIR = PROJECT_ROOT / "output"
TRAINING_LOG_FILE = OUTPUT_DIR / "training.log"


# ##################################################################
# ensure local directory
# creates the local directory if it does not exist
def ensure_local_directory() -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)


# ##################################################################
# ensure output directory
# creates the output directory if it does not exist
def ensure_output_directory() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ##################################################################
# training log
# writes a message to the training log file with timestamp
def training_log(message: str) -> None:
    ensure_output_directory()
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}\n"
    with open(TRAINING_LOG_FILE, "a") as f:
        f.write(line)
    print(message)


# ##################################################################
# configuration
# device detection and constants for the ranking system
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DB_FILE = str(LOCAL_DIR / "news_ranker.db")
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
EMBEDDING_DIM = 768
EMBEDDING_MODEL_VERSION = 3
CHUNK_SIZE_WORDS = 500
MAX_SEQ_LENGTH = 512
CORRECTION_EPOCHS = 50  # Epochs added per batch of corrections
RETRAIN_EPOCHS = 500  # Epochs added by retrain command
MAX_EPOCHS = 2000  # Cap on pending epochs
SAVE_INTERVAL_EPOCHS = 50  # Save model every N epochs
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.1
HIDDEN_DIM = 128
ATTENTION_DIM = 128
SIMILARITY_THRESHOLD = 0.92
MIN_SCORE = 1.0
MAX_SCORE = 10.0
DEFAULT_SCORE = 5.5
SERVER_PORT = 19091

# Worker pool configuration
WORKER_POOL_SIZE = 6
WORKER_MAX_AGE_SECONDS = 600  # 10 minutes
WORKER_CYCLE_INTERVAL_SECONDS = 60  # Check for old workers every minute
RANKING_TIMEOUT_SECONDS = 30


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
            cleaned_html_text TEXT,
            embedding BLOB,
            current_rank REAL,
            user_target_rank REAL,
            is_training_data INTEGER DEFAULT 0
        )
    ''')
    ensure_articles_schema(conn)
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
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            predicted_score REAL NOT NULL,
            corrected_score REAL NOT NULL,
            delta REAL NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT DEFAULT CURRENT_TIMESTAMP,
            completed_at TEXT,
            training_type TEXT NOT NULL,
            samples_count INTEGER,
            epochs_run INTEGER,
            final_loss REAL,
            best_loss REAL,
            status TEXT DEFAULT 'running'
        )
    ''')
    conn.commit()
    conn.close()
    check_model_version_migration()
    cleanup_stale_training_records()


# ##################################################################
# ensure training history schema
# adds pid column if missing
def ensure_training_history_schema() -> None:
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(training_history)")
    columns = {row[1] for row in cursor.fetchall()}
    if "pid" not in columns:
        cursor.execute("ALTER TABLE training_history ADD COLUMN pid INTEGER")
        conn.commit()
    conn.close()


# ##################################################################
# cleanup stale training records
# marks any 'running' training records as 'interrupted' on startup
# any record from a different pid or any running record is stale since we just started
def cleanup_stale_training_records() -> None:
    ensure_training_history_schema()
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE training_history SET status = 'interrupted', completed_at = CURRENT_TIMESTAMP "
        "WHERE status = 'running'"
    )
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    if affected > 0:
        print(f"Cleaned up {affected} stale training record(s) from previous run.")


# ##################################################################
# ensure articles schema
# adds missing columns to the articles table
def ensure_articles_schema(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(articles)")
    columns = {row[1] for row in cursor.fetchall()}
    if "cleaned_html_text" not in columns:
        cursor.execute("ALTER TABLE articles ADD COLUMN cleaned_html_text TEXT")
        conn.commit()


# ##################################################################
# get database connection
# returns a new sqlite connection for thread safety
def get_database_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_FILE)


# ##################################################################
# load setting
# fetches a value from the settings table
def load_setting(key: str) -> Optional[str]:
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


# ##################################################################
# save setting
# stores a value in the settings table
def save_setting(key: str, value: str) -> None:
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
        (key, value)
    )
    conn.commit()
    conn.close()


# ##################################################################
# compute calibration params
# computes linear scale and bias for min/max mapping
def compute_calibration_params(
    min_pred: float,
    max_pred: float,
    min_target: float,
    max_target: float
) -> tuple[float, float]:
    if abs(max_pred - min_pred) < 1e-6 or abs(max_target - min_target) < 1e-6:
        return 1.0, 0.0
    scale = (max_target - min_target) / (max_pred - min_pred)
    bias = min_target - min_pred * scale
    return scale, bias


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
            "DELETE FROM settings WHERE key IN ('calibration_scale', 'calibration_bias')"
        )
        cursor.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES ('embedding_model_version', ?)",
            (str(EMBEDDING_MODEL_VERSION),)
        )
        conn.commit()
        print("Old embeddings and trained model cleared. Run fetch-training and retrain.")
    conn.close()


# ##################################################################
# record correction
# stores a correction event with delta for accuracy tracking
def record_correction(url: str, predicted_score: float, corrected_score: float) -> None:
    delta = corrected_score - predicted_score
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO corrections (url, predicted_score, corrected_score, delta) VALUES (?, ?, ?, ?)",
        (url, predicted_score, corrected_score, delta)
    )
    conn.commit()
    conn.close()


# ##################################################################
# start training record
# creates a new training history entry and returns its id
def start_training_record(training_type: str, samples_count: int) -> Optional[int]:
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO training_history (training_type, samples_count, status, pid) VALUES (?, ?, 'running', ?)",
        (training_type, samples_count, os.getpid())
    )
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id


# ##################################################################
# complete training record
# updates a training history entry with completion info
def complete_training_record(
    record_id: int,
    epochs_run: int,
    final_loss: float,
    best_loss: Optional[float] = None,
    status: str = "completed"
) -> None:
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute(
        """UPDATE training_history
           SET completed_at = CURRENT_TIMESTAMP,
               epochs_run = ?,
               final_loss = ?,
               best_loss = ?,
               status = ?
           WHERE id = ?""",
        (epochs_run, final_loss, best_loss or final_loss, status, record_id)
    )
    conn.commit()
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
# hierarchical attention ranker
# HAN-style attention pooling over sentence and chunk embeddings
class HierarchicalAttentionRanker(nn.Module):

    # ##################################################################
    # init
    # sets up attention pooling and ranking head
    def __init__(self, input_dim: int) -> None:
        super(HierarchicalAttentionRanker, self).__init__()
        self.sentence_attention = nn.Linear(input_dim, ATTENTION_DIM)
        self.sentence_context = nn.Parameter(torch.randn(ATTENTION_DIM))
        self.user_sentence_query = nn.Linear(input_dim, ATTENTION_DIM)
        self.chunk_attention = nn.Linear(input_dim, ATTENTION_DIM)
        self.chunk_context = nn.Parameter(torch.randn(ATTENTION_DIM))
        self.user_chunk_query = nn.Linear(input_dim, ATTENTION_DIM)
        self.ranking_head = RankingHead(input_dim)

    # ##################################################################
    # attention pool
    # applies attention pooling over a sequence of embeddings
    def attention_pool(
        self,
        embeddings: torch.Tensor,
        attention_layer: nn.Linear,
        context_vector: nn.Parameter
    ) -> torch.Tensor:
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        scores = torch.tanh(attention_layer(embeddings))
        weights = torch.softmax(torch.matmul(scores, context_vector), dim=0)
        return torch.sum(embeddings * weights.unsqueeze(1), dim=0)

    # ##################################################################
    # encode document
    # computes document vector from hierarchical embeddings
    def encode_document(
        self,
        chunks: list[torch.Tensor],
        user_profile: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        if not chunks:
            doc_vector = torch.zeros(EMBEDDING_DIM, device=device)
        else:
            if user_profile is None:
                sentence_context = self.sentence_context
                chunk_context = self.chunk_context
            else:
                user_profile = user_profile.to(device)
                sentence_context = self.user_sentence_query(user_profile)
                chunk_context = self.user_chunk_query(user_profile)

            chunk_vectors = []
            for chunk in chunks:
                if chunk.numel() == 0:
                    continue
                chunk = chunk.to(device)
                chunk_vectors.append(self.attention_pool(chunk, self.sentence_attention, sentence_context))

            if not chunk_vectors:
                doc_vector = torch.zeros(EMBEDDING_DIM, device=device)
            else:
                chunk_matrix = torch.stack(chunk_vectors)
                doc_vector = self.attention_pool(chunk_matrix, self.chunk_attention, chunk_context)

        return doc_vector

    # ##################################################################
    # forward
    # computes document score from hierarchical embeddings
    def forward(
        self,
        chunks: list[torch.Tensor],
        user_profile: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        doc_vector = self.encode_document(chunks, user_profile)
        return self.ranking_head(doc_vector.unsqueeze(0)).squeeze(0)


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
        self.ranker = HierarchicalAttentionRanker(EMBEDDING_DIM).to(DEVICE)
        self.calibration_scale = self.load_calibration_setting("calibration_scale", 1.0)
        self.calibration_bias = self.load_calibration_setting("calibration_bias", 0.0)
        saved_weights = load_model_weights("ranker")
        if saved_weights:
            self.ranker.load_state_dict(saved_weights)
            print("Loaded trained model weights from database.")
        else:
            print("No saved weights found, using fresh model.")
        self.ranker.eval()
        self.is_training = False
        self._training_lock = threading.Lock()  # Serialize training only
        self._embedder_lock = threading.Lock()  # SentenceTransformer not thread-safe
        # Cached training data for fast prediction (refreshed after training)
        self._cached_samples: Optional[list[list[torch.Tensor]]] = None
        self._cached_targets: Optional[torch.Tensor] = None
        self._cached_user_profile: Optional[torch.Tensor] = None
        self._cached_doc_vectors: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None
        self._refresh_prediction_cache()

    # ##################################################################
    # refresh prediction cache
    # loads training data and precomputes user profile and doc vectors
    def _refresh_prediction_cache(self) -> None:
        rows = self.load_training_data()
        if not rows:
            self._cached_samples = None
            self._cached_targets = None
            self._cached_user_profile = None
            self._cached_doc_vectors = None
            return
        samples, targets = self.prepare_training_samples(rows)
        if not samples:
            self._cached_samples = None
            self._cached_targets = None
            self._cached_user_profile = None
            self._cached_doc_vectors = None
            return
        self._cached_samples = samples
        self._cached_targets = targets
        self._cached_user_profile = self.compute_user_profile_from_samples(samples, targets)
        # Precompute document vectors for similarity lookup
        doc_vectors = []
        for sample, target in zip(samples, targets):
            vec = self.compute_document_mean_vector(sample)
            if vec is not None:
                doc_vectors.append((vec, target))
        self._cached_doc_vectors = doc_vectors if doc_vectors else None

    # ##################################################################
    # split into sentences
    # splits text into sentence-like segments
    def split_into_sentences(self, text: str) -> list[str]:
        cleaned = re.sub(r'\s+', ' ', text).strip()
        if not cleaned:
            return []
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    # ##################################################################
    # chunk sentences
    # groups sentences into chunks based on word count
    def chunk_sentences(self, sentences: list[str]) -> list[list[str]]:
        chunks = []
        current = []
        current_words = 0

        for sentence in sentences:
            word_count = len(sentence.split())
            if current and current_words + word_count > CHUNK_SIZE_WORDS:
                chunks.append(current)
                current = []
                current_words = 0
            current.append(sentence)
            current_words += word_count

        if current:
            chunks.append(current)
        return chunks

    # ##################################################################
    # build hierarchical embeddings
    # returns list of sentence-embedding tensors per chunk
    def build_hierarchical_embeddings(self, text: str, to_cpu: bool = True) -> list[torch.Tensor]:
        sentences = self.split_into_sentences(text)
        if not sentences:
            return []

        chunks = self.chunk_sentences(sentences)
        flat_sentences = [sentence for chunk in chunks for sentence in chunk]
        sizes = [len(chunk) for chunk in chunks]
        if not flat_sentences:
            return []

        with torch.no_grad():
            with self._embedder_lock:  # SentenceTransformer not thread-safe
                embeddings = self.embedder.encode(flat_sentences, convert_to_tensor=True, show_progress_bar=False)

        chunk_tensors = []
        start = 0
        for size in sizes:
            end = start + size
            chunk = embeddings[start:end]
            chunk_tensors.append(chunk.cpu() if to_cpu else chunk)
            start = end

        return chunk_tensors

    # ##################################################################
    # compute document mean vector
    # averages sentence embeddings for a document
    def compute_document_mean_vector(self, chunks: list[torch.Tensor]) -> Optional[torch.Tensor]:
        if not chunks:
            return None
        sentence_tensors = [chunk for chunk in chunks if chunk.numel() > 0]
        if not sentence_tensors:
            return None
        sentence_matrix = torch.cat(sentence_tensors, dim=0)
        if sentence_matrix.numel() == 0:
            return None
        return torch.mean(sentence_matrix, dim=0)

    # ##################################################################
    # load calibration setting
    # loads a float calibration value from settings
    def load_calibration_setting(self, key: str, default: float) -> float:
        value = load_setting(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default

    # ##################################################################
    # compute user profile from samples
    # builds a weighted profile vector from hierarchical embeddings
    def compute_user_profile_from_samples(
        self,
        samples: list[list[torch.Tensor]],
        targets: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if targets.dim() == 2:
            targets = targets.squeeze(1)

        if not samples or targets.numel() == 0:
            return None

        vectors = []
        weights = []
        with torch.no_grad():
            for chunks, target in zip(samples, targets):
                if not chunks:
                    continue
                sentence_matrix = torch.cat([chunk for chunk in chunks if chunk.numel() > 0], dim=0)
                if sentence_matrix.numel() == 0:
                    continue
                doc_vector = torch.mean(sentence_matrix, dim=0)
                weight = (target - DEFAULT_SCORE) / (MAX_SCORE - MIN_SCORE)
                if torch.isclose(weight, torch.tensor(0.0, device=weight.device)):
                    continue
                vectors.append(doc_vector)
                weights.append(weight)

            if not vectors:
                return None

            weight_tensor = torch.stack(weights)
            denom = torch.sum(torch.abs(weight_tensor)).clamp(min=1e-6)
            stacked = torch.stack(vectors)
            profile = torch.sum(stacked * weight_tensor.unsqueeze(1), dim=0) / denom
            return profile

    # ##################################################################
    # compute user profile from database
    # loads training data and computes profile vector
    def compute_user_profile_from_db(self) -> Optional[torch.Tensor]:
        rows = self.load_training_data()
        samples, targets = self.prepare_training_samples(rows)
        if not samples:
            return None
        return self.compute_user_profile_from_samples(samples, targets)

    # ##################################################################
    # update calibration
    # computes and stores calibration from training samples
    def update_calibration(
        self,
        ranker: HierarchicalAttentionRanker,
        samples: list[list[torch.Tensor]],
        targets: torch.Tensor,
        user_profile: Optional[torch.Tensor]
    ) -> None:
        if not samples or targets.numel() == 0:
            self.calibration_scale = 1.0
            self.calibration_bias = 0.0
            save_setting("calibration_scale", str(self.calibration_scale))
            save_setting("calibration_bias", str(self.calibration_bias))
            return

        with torch.no_grad():
            preds = torch.stack([ranker(chunks, user_profile=user_profile) for chunks in samples]).squeeze(1)

        min_pred = preds.min().item()
        max_pred = preds.max().item()
        min_target = targets.min().item()
        max_target = targets.max().item()
        scale, bias = compute_calibration_params(min_pred, max_pred, min_target, max_target)
        self.calibration_scale = scale
        self.calibration_bias = bias
        save_setting("calibration_scale", str(scale))
        save_setting("calibration_bias", str(bias))

    # ##################################################################
    # clamp score
    # ensures score falls within valid range
    def clamp_score(self, score: float) -> float:
        return max(MIN_SCORE, min(MAX_SCORE, score))

    # ##################################################################
    # predict score
    # runs inference to get a clamped score for text
    def predict_score(self, text: str) -> float:
        # Snapshot shared state at start (atomic under GIL)
        ranker = self.ranker
        scale = self.calibration_scale
        bias = self.calibration_bias
        user_profile = self._cached_user_profile
        doc_vectors = self._cached_doc_vectors

        with torch.no_grad():
            chunks = self.build_hierarchical_embeddings(text, to_cpu=False)

            # Use snapshot ranker with cached user profile
            raw_score = ranker(chunks, user_profile=user_profile).item()
            calibrated = raw_score * scale + bias

            # Similarity adjustment using cached doc vectors
            if doc_vectors:
                doc_vector = self.compute_document_mean_vector(chunks)
                if doc_vector is not None:
                    train_matrix = torch.stack([vec.to(DEVICE) for vec, _ in doc_vectors])
                    doc_vector = doc_vector.to(DEVICE)
                    sims = F.cosine_similarity(train_matrix, doc_vector.unsqueeze(0), dim=1)
                    max_sim, max_idx = torch.max(sims, dim=0)
                    if max_sim.item() >= SIMILARITY_THRESHOLD:
                        alpha = (max_sim.item() - SIMILARITY_THRESHOLD) / (1.0 - SIMILARITY_THRESHOLD)
                        target_value = doc_vectors[max_idx][1].item()
                        calibrated = (1.0 - alpha) * calibrated + alpha * target_value

            return self.clamp_score(calibrated)

    # ##################################################################
    # get embedding bytes
    # returns pickled embedding for database storage
    def get_embedding_bytes(self, text: str) -> Optional[bytes]:
        with torch.no_grad():
            chunks = self.build_hierarchical_embeddings(text, to_cpu=True)
            if not chunks:
                return None
            return pickle.dumps(chunks)

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
    # prepare training samples
    # deserializes hierarchical embeddings and prepares tensors
    def prepare_training_samples(self, rows: list[tuple]) -> tuple[list[list[torch.Tensor]], torch.Tensor]:
        embedding_list = []
        target_list = []
        for embedding_blob, target in rows:
            if not embedding_blob:
                continue
            chunks = pickle.loads(embedding_blob)
            if not chunks:
                continue
            embedding_list.append([chunk.to(DEVICE) for chunk in chunks])
            target_list.append(target)
        y_train = torch.tensor(target_list, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        return embedding_list, y_train

    # ##################################################################
    # create new ranker
    # creates a copy of the current ranker for training
    def create_new_ranker(self) -> HierarchicalAttentionRanker:
        new_ranker = HierarchicalAttentionRanker(EMBEDDING_DIM).to(DEVICE)
        new_ranker.load_state_dict(self.ranker.state_dict())
        new_ranker.train()
        return new_ranker

    # ##################################################################
    # generate missing embeddings
    # creates embeddings for training data that lacks them
    def generate_missing_embeddings(self) -> int:
        conn = get_database_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT url, extracted_text, cleaned_html_text FROM articles "
            "WHERE user_target_rank IS NOT NULL AND embedding IS NULL"
        )
        rows = cursor.fetchall()

        count = 0
        for url, extracted_text, cleaned_html_text in rows:
            text_for_embedding = extracted_text or cleaned_html_text
            if not text_for_embedding:
                print(f"  Skipping {url[:50]}... (no text)")
                continue
            embedding_blob = self.get_embedding_bytes(text_for_embedding)
            if not embedding_blob:
                print(f"  Skipping {url[:50]}... (no embeddings)")
                continue
            cursor.execute("UPDATE articles SET embedding = ? WHERE url = ?", (embedding_blob, url))
            count += 1
            print(f"  Generated embedding for {url[:50]}...")

        conn.commit()
        conn.close()
        return count

    # ##################################################################
    # retrain
    # synchronous training for tests and CLI - trains for fixed epochs
    def retrain(self, epochs: int = 100) -> None:
        with self._training_lock:
            if self.is_training:
                training_log("Training already in progress. Skipping.")
                return
            self.is_training = True

        try:
            rows = self.load_training_data()
            if not rows:
                training_log("No training data found.")
                return

            x_train, y_train = self.prepare_training_samples(rows)
            if not x_train:
                training_log("No valid training samples after embedding load.")
                return

            ranker = self.create_new_ranker()
            user_profile = self.compute_user_profile_from_samples(x_train, y_train)
            optimizer = torch.optim.Adam(ranker.parameters(), lr=LEARNING_RATE)
            criterion = nn.MSELoss()

            best_loss = float("inf")
            best_state = None
            for _ in range(epochs):
                optimizer.zero_grad()
                outputs = torch.stack([ranker(chunks, user_profile=user_profile) for chunks in x_train])
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_state = {k: v.clone() for k, v in ranker.state_dict().items()}

            if best_state:
                ranker.load_state_dict(best_state)
            ranker.eval()
            self.update_calibration(ranker, x_train, y_train, user_profile)
            self.ranker = ranker
            save_model_weights("ranker", self.ranker.state_dict())
            self._refresh_prediction_cache()
        finally:
            self.is_training = False


# ##################################################################
# ranking request dataclass
# represents a request in the ranking queue
@dataclass
class RankingRequest:
    request_id: str
    text: str


# ##################################################################
# ranking response dataclass
# represents a response from a worker
@dataclass
class RankingResponse:
    request_id: str
    score: float
    error: Optional[str] = None


# ##################################################################
# ranking worker
# independent worker with its own embedder for parallel ranking
class RankingWorker:

    def __init__(self, worker_id: int, shared_state: dict) -> None:
        self.worker_id = worker_id
        self.shared_state = shared_state  # Contains ranker, calibration, cached data
        self.created_at = time.time()
        self.shutdown_requested = False  # Graceful shutdown flag
        print(f"Worker {worker_id}: Loading embedder...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
        self.embedder.max_seq_length = MAX_SEQ_LENGTH
        print(f"Worker {worker_id}: Ready")

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > WORKER_MAX_AGE_SECONDS

    def request_shutdown(self) -> None:
        self.shutdown_requested = True

    def split_into_sentences(self, text: str) -> list[str]:
        cleaned = re.sub(r'\s+', ' ', text).strip()
        if not cleaned:
            return []
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def chunk_sentences(self, sentences: list[str]) -> list[list[str]]:
        chunks = []
        current = []
        current_words = 0
        for sentence in sentences:
            word_count = len(sentence.split())
            if current and current_words + word_count > CHUNK_SIZE_WORDS:
                chunks.append(current)
                current = []
                current_words = 0
            current.append(sentence)
            current_words += word_count
        if current:
            chunks.append(current)
        return chunks

    def build_hierarchical_embeddings(self, text: str) -> list[torch.Tensor]:
        sentences = self.split_into_sentences(text)
        if not sentences:
            return []
        chunks = self.chunk_sentences(sentences)
        flat_sentences = [sentence for chunk in chunks for sentence in chunk]
        sizes = [len(chunk) for chunk in chunks]
        if not flat_sentences:
            return []
        with torch.no_grad():
            # Each worker has its own embedder - no lock needed!
            embeddings = self.embedder.encode(flat_sentences, convert_to_tensor=True, show_progress_bar=False)
        chunk_tensors = []
        start = 0
        for size in sizes:
            end = start + size
            chunk_tensors.append(embeddings[start:end])
            start = end
        return chunk_tensors

    def compute_document_mean_vector(self, chunks: list[torch.Tensor]) -> Optional[torch.Tensor]:
        if not chunks:
            return None
        sentence_tensors = [chunk for chunk in chunks if chunk.numel() > 0]
        if not sentence_tensors:
            return None
        sentence_matrix = torch.cat(sentence_tensors, dim=0)
        if sentence_matrix.numel() == 0:
            return None
        return torch.mean(sentence_matrix, dim=0)

    def predict_score(self, text: str) -> float:
        # Snapshot shared state (atomic under GIL)
        ranker = self.shared_state.get("ranker")
        scale = self.shared_state.get("calibration_scale", 1.0)
        bias = self.shared_state.get("calibration_bias", 0.0)
        user_profile = self.shared_state.get("user_profile")
        doc_vectors = self.shared_state.get("doc_vectors")

        with torch.no_grad():
            chunks = self.build_hierarchical_embeddings(text)
            if not chunks:
                return DEFAULT_SCORE

            raw_score = ranker(chunks, user_profile=user_profile).item()
            calibrated = raw_score * scale + bias

            if doc_vectors:
                doc_vector = self.compute_document_mean_vector(chunks)
                if doc_vector is not None:
                    train_matrix = torch.stack([vec.to(DEVICE) for vec, _ in doc_vectors])
                    doc_vector = doc_vector.to(DEVICE)
                    sims = F.cosine_similarity(train_matrix, doc_vector.unsqueeze(0), dim=1)
                    max_sim, max_idx = torch.max(sims, dim=0)
                    if max_sim.item() >= SIMILARITY_THRESHOLD:
                        alpha = (max_sim.item() - SIMILARITY_THRESHOLD) / (1.0 - SIMILARITY_THRESHOLD)
                        target_value = doc_vectors[max_idx][1].item()
                        calibrated = (1.0 - alpha) * calibrated + alpha * target_value

            return max(MIN_SCORE, min(MAX_SCORE, calibrated))


# ##################################################################
# worker pool
# manages a pool of ranking workers for parallel processing
class WorkerPool:

    def __init__(self, engine: ScoringEngine) -> None:
        self.engine = engine
        self.request_queue: queue.Queue[RankingRequest] = queue.Queue()
        self.response_map: dict[str, RankingResponse] = {}
        self.response_lock = threading.Lock()
        self.response_events: dict[str, threading.Event] = {}
        self.workers: list[tuple[RankingWorker, threading.Thread]] = []
        self.running = True
        self.shared_state = self._build_shared_state()
        self._start_workers()

    def _build_shared_state(self) -> dict:
        return {
            "ranker": self.engine.ranker,
            "calibration_scale": self.engine.calibration_scale,
            "calibration_bias": self.engine.calibration_bias,
            "user_profile": self.engine._cached_user_profile,
            "doc_vectors": self.engine._cached_doc_vectors,
        }

    def refresh_shared_state(self) -> None:
        # Called after training completes - update in-place so workers see changes
        new_state = self._build_shared_state()
        self.shared_state.update(new_state)
        training_log(f"Model pushed to {len(self.workers)} workers")

    def _start_workers(self) -> None:
        for i in range(WORKER_POOL_SIZE):
            self._spawn_worker(i)

    def _spawn_worker(self, worker_id: int) -> None:
        worker = RankingWorker(worker_id, self.shared_state)
        thread = threading.Thread(target=self._worker_loop, args=(worker,), daemon=True)
        thread.start()
        self.workers.append((worker, thread))

    def _worker_loop(self, worker: RankingWorker) -> None:
        while self.running:
            # Check for graceful shutdown BETWEEN tasks (not during)
            if worker.shutdown_requested:
                print(f"Worker {worker.worker_id}: Graceful shutdown after {time.time() - worker.created_at:.0f}s")
                break

            try:
                request = self.request_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Process the task to completion before checking shutdown
            try:
                score = worker.predict_score(request.text)
                response = RankingResponse(request_id=request.request_id, score=score)
            except Exception as e:
                response = RankingResponse(request_id=request.request_id, score=0.0, error=str(e))

            with self.response_lock:
                self.response_map[request.request_id] = response
                if request.request_id in self.response_events:
                    self.response_events[request.request_id].set()

        print(f"Worker {worker.worker_id}: Exited")

    def submit_request(self, text: str) -> str:
        request_id = str(uuid.uuid4())
        event = threading.Event()
        with self.response_lock:
            self.response_events[request_id] = event
        self.request_queue.put(RankingRequest(request_id=request_id, text=text))
        return request_id

    def get_response(self, request_id: str, timeout: float) -> Optional[RankingResponse]:
        event = self.response_events.get(request_id)
        if not event:
            return None
        if not event.wait(timeout=timeout):
            return None
        with self.response_lock:
            response = self.response_map.pop(request_id, None)
            self.response_events.pop(request_id, None)
        return response

    def cycle_workers(self) -> None:
        # Clean up dead workers from the list
        self.workers = [(w, t) for w, t in self.workers if t.is_alive()]
        alive_count = len(self.workers)

        # Request graceful shutdown for the oldest expired worker (one per cycle)
        # Only if we're at or above target (don't request shutdown if understaffed)
        if alive_count >= WORKER_POOL_SIZE:
            expired_workers = [(w, t) for w, t in self.workers
                               if w.is_expired() and not w.shutdown_requested]
            if expired_workers:
                oldest_worker = min(expired_workers, key=lambda wt: wt[0].created_at)[0]
                print(f"Worker {oldest_worker.worker_id}: Requesting shutdown (age={time.time() - oldest_worker.created_at:.0f}s)")
                oldest_worker.request_shutdown()

        # Spawn new workers if we're below target
        # This happens naturally as workers shut down gracefully
        while alive_count < WORKER_POOL_SIZE:
            new_id = max((w.worker_id for w, _ in self.workers), default=-1) + 1
            self._spawn_worker(new_id)
            alive_count += 1

    def shutdown(self) -> None:
        self.running = False
        for _, thread in self.workers:
            thread.join(timeout=5.0)


# ##################################################################
# training loop
# continuous training with epoch-by-epoch processing and corrections queue
class TrainingLoop:

    def __init__(self, engine: ScoringEngine, worker_pool: "WorkerPool") -> None:
        self.engine = engine
        self.worker_pool = worker_pool
        self.epochs_queue: queue.Queue[int] = queue.Queue()
        self.running = True
        self._lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def add_epochs(self, count: int) -> None:
        """Add epochs to the training queue (will be capped at MAX_EPOCHS)"""
        self.epochs_queue.put(count)

    def _run(self) -> None:
        epochs_remaining = 0
        ranker = None
        optimizer = None
        x_train = None
        y_train = None
        user_profile = None
        best_loss = float("inf")
        best_state = None
        epoch_count = 0
        last_log_time = time.time()
        last_save_epoch = 0
        record_id = None
        samples_count = 0

        while self.running:
            # 1. Collect any new epoch requests
            while True:
                try:
                    new_epochs = self.epochs_queue.get_nowait()
                    epochs_remaining = min(epochs_remaining + new_epochs, MAX_EPOCHS)
                except queue.Empty:
                    break

            # 2. If no epochs remaining, sleep and continue
            if epochs_remaining <= 0:
                time.sleep(0.5)
                continue

            # 3. Initialize or reload training if needed
            if ranker is None:
                # Generate any missing embeddings first
                missing = self.engine.generate_missing_embeddings()
                if missing > 0:
                    training_log(f"Generated {missing} missing embeddings")

                rows = self.engine.load_training_data()
                rows = [(emb, target) for emb, target in rows if emb is not None]
                if not rows:
                    training_log("No training data with embeddings found.")
                    epochs_remaining = 0
                    continue

                x_train, y_train = self.engine.prepare_training_samples(rows)
                if not x_train:
                    training_log("No valid training samples after embedding load.")
                    epochs_remaining = 0
                    continue

                samples_count = len(rows)
                ranker = self.engine.create_new_ranker()
                user_profile = self.engine.compute_user_profile_from_samples(x_train, y_train)
                optimizer = torch.optim.Adam(ranker.parameters(), lr=LEARNING_RATE)
                best_loss = float("inf")
                best_state = None
                epoch_count = 0
                last_save_epoch = 0
                record_id = start_training_record("continuous", samples_count)
                training_log(f"=== TRAINING START ({samples_count} samples, {epochs_remaining} epochs queued) ===")

            # 4. Do one epoch
            criterion = nn.MSELoss()
            optimizer.zero_grad()
            outputs = torch.stack([ranker(chunks, user_profile=user_profile) for chunks in x_train])
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            epoch_count += 1
            epochs_remaining -= 1

            if current_loss < best_loss:
                best_loss = current_loss
                best_state = {k: v.clone() for k, v in ranker.state_dict().items()}

            # 5. Log every minute
            now = time.time()
            if now - last_log_time >= 60:
                avg_error = best_loss ** 0.5
                training_log(f"  epoch {epoch_count}, {epochs_remaining} remaining, avg error: ±{avg_error:.2f} pts")
                last_log_time = now

            # 6. Save periodically
            if epoch_count - last_save_epoch >= SAVE_INTERVAL_EPOCHS:
                self._save_model(ranker, best_state, x_train, y_train, user_profile)
                last_save_epoch = epoch_count

            # 7. If done with current batch, finalize
            if epochs_remaining <= 0:
                self._save_model(ranker, best_state, x_train, y_train, user_profile)
                avg_error = best_loss ** 0.5
                training_log(f"=== TRAINING COMPLETE ({epoch_count} epochs, avg error: ±{avg_error:.2f} pts) ===")
                if record_id:
                    complete_training_record(record_id, epoch_count, best_loss, best_loss)
                ranker = None  # Reset for next training batch
                record_id = None

    def _save_model(
        self,
        ranker: HierarchicalAttentionRanker,
        best_state: Optional[dict],
        x_train: list,
        y_train: torch.Tensor,
        user_profile: Optional[torch.Tensor]
    ) -> None:
        if best_state:
            ranker.load_state_dict(best_state)
        ranker.eval()
        self.engine.update_calibration(ranker, x_train, y_train, user_profile)
        self.engine.ranker = ranker
        save_model_weights("ranker", ranker.state_dict())
        self.engine._refresh_prediction_cache()
        self.worker_pool.refresh_shared_state()
        ranker.train()

    def shutdown(self) -> None:
        self.running = False
        self.thread.join(timeout=5.0)


# Global pool and training loop
worker_pool: Optional[WorkerPool] = None
training_loop: Optional[TrainingLoop] = None


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


# ##################################################################
# training set
# returns all trained URLs and their user target scores
def training_set() -> list[dict]:
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT url, user_target_rank FROM articles WHERE user_target_rank IS NOT NULL"
    )
    rows = cursor.fetchall()
    conn.close()
    return [{"url": row[0], "score": row[1]} for row in rows]


# global engine instance
engine: Optional[ScoringEngine] = None


# ##################################################################
RETRAIN_INTERVAL_SECONDS = 2 * 60 * 60


# ##################################################################
# scheduled retrain task
# runs retrain every 2 hours in the background via training loop
async def scheduled_retrain_task() -> None:
    while True:
        await asyncio.sleep(RETRAIN_INTERVAL_SECONDS)
        print("Scheduled retrain starting...")
        if training_loop:
            training_loop.add_epochs(RETRAIN_EPOCHS)


# ##################################################################
# worker cycling task
# cycles out oldest worker every minute to keep workers fresh
async def worker_cycling_task() -> None:
    while True:
        await asyncio.sleep(WORKER_CYCLE_INTERVAL_SECONDS)
        if worker_pool:
            worker_pool.cycle_workers()


# ##################################################################
# lifespan
# manages startup and shutdown of the fastapi application
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, worker_pool, training_loop
    initialize_database()
    engine = ScoringEngine()
    worker_pool = WorkerPool(engine)
    training_loop = TrainingLoop(engine, worker_pool)
    retrain_task = asyncio.create_task(scheduled_retrain_task())
    cycling_task = asyncio.create_task(worker_cycling_task())
    print(f"System initialized with {WORKER_POOL_SIZE} workers. Scheduled retrain every 2 hours.")
    yield
    retrain_task.cancel()
    cycling_task.cancel()
    training_loop.shutdown()
    worker_pool.shutdown()
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
# clean html text
# removes extra whitespace and normalizes text extracted from html
def clean_html_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ##################################################################
# normalize text
# trims and collapses empty strings to none
def normalize_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    stripped = text.strip()
    return stripped if stripped else None


# ##################################################################
# fetch with trafilatura data
# returns extracted article text and cleaned html text
def fetch_with_trafilatura_data(url: str) -> tuple[Optional[str], Optional[str]]:
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return None, None
        downloaded_text = (
            downloaded.decode("utf-8", errors="ignore")
            if isinstance(downloaded, bytes)
            else downloaded
        )
        extracted = trafilatura.extract(downloaded_text)
        cleaned_html = clean_html_text(downloaded_text)
        return normalize_text(extracted), normalize_text(cleaned_html)
    except Exception:
        return None, None


# ##################################################################
# fetch with trafilatura
# fast extraction for static html pages
def fetch_with_trafilatura(url: str) -> Optional[str]:
    extracted, _ = fetch_with_trafilatura_data(url)
    return extracted


# ##################################################################
# fetch with playwright data
# returns extracted article text and cleaned html text
def fetch_with_playwright_data(url: str) -> tuple[Optional[str], Optional[str]]:
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
        cleaned_html = clean_html_text(content)
        return normalize_text(extracted), normalize_text(cleaned_html)
    except Exception:
        return None, None


# ##################################################################
# fetch with playwright
# uses headless browser for javascript-rendered pages
def fetch_with_playwright(url: str) -> Optional[str]:
    extracted, cleaned_html = fetch_with_playwright_data(url)
    if extracted:
        return extracted
    return cleaned_html if cleaned_html and len(cleaned_html) > 100 else None


# ##################################################################
# fetch article texts
# tries trafilatura first, falls back to playwright for js-rendered pages
async def fetch_article_texts(url: str) -> tuple[Optional[str], Optional[str]]:
    loop = asyncio.get_event_loop()

    try:
        extracted, cleaned_html = await asyncio.wait_for(
            loop.run_in_executor(None, fetch_with_trafilatura_data, url),
            timeout=15
        )
    except asyncio.TimeoutError:
        extracted, cleaned_html = None, None

    if extracted and len(extracted) > 100:
        return extracted, cleaned_html

    try:
        extracted_pw, cleaned_html_pw = await asyncio.wait_for(
            loop.run_in_executor(None, fetch_with_playwright_data, url),
            timeout=45
        )
    except asyncio.TimeoutError:
        extracted_pw, cleaned_html_pw = None, None

    if extracted_pw and len(extracted_pw) > 100:
        return extracted_pw, cleaned_html_pw or cleaned_html

    return None, cleaned_html_pw or cleaned_html


# ##################################################################
# fetch article text
# returns extracted text with cleaned html fallback
async def fetch_article_text(url: str) -> Optional[str]:
    extracted, cleaned_html = await fetch_article_texts(url)
    if extracted:
        return extracted
    return cleaned_html if cleaned_html and len(cleaned_html) > 100 else None


# ##################################################################
# rank article endpoint
# fetches and scores url fresh every time, always returns 200
@app.get("/rank", response_model=RankResponse)
async def rank_article(url: str = Query(..., min_length=1)):
    try:
        extracted_text = await fetch_article_text(url)
        if not extracted_text:
            return {"url": url, "rank": 0.0, "source": "unavailable"}

        # Submit to worker pool and wait for response
        if not worker_pool:
            return {"url": url, "rank": 0.0, "source": "error"}

        request_id = worker_pool.submit_request(extracted_text)
        # Wait in executor to not block event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, worker_pool.get_response, request_id, RANKING_TIMEOUT_SECONDS
        )

        if response is None:
            return {"url": url, "rank": 0.0, "source": "timeout"}
        if response.error:
            return {"url": url, "rank": 0.0, "source": "error"}
        return {"url": url, "rank": response.score, "source": "ok"}
    except Exception:
        return {"url": url, "rank": 0.0, "source": "error"}


# ##################################################################
# get article state
# returns whether article exists plus stored texts and embedding
def get_article_state(url: str) -> tuple[bool, Optional[str], Optional[str], Optional[bytes]]:
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT extracted_text, cleaned_html_text, embedding FROM articles WHERE url = ?",
        (url,)
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return True, row[0], row[1], row[2]
    return False, None, None, None


# ##################################################################
# store new article with target
# inserts a new article with user feedback
def store_new_article_with_target(
    url: str,
    extracted_text: Optional[str],
    cleaned_html_text: Optional[str],
    embedding_blob: Optional[bytes],
    target_score: float
) -> None:
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO articles (url, extracted_text, cleaned_html_text, embedding, user_target_rank)
        VALUES (?, ?, ?, ?, ?)
    ''', (url, extracted_text, cleaned_html_text, embedding_blob, target_score))
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
# update article texts
# updates text fields and embedding without overwriting existing data
def update_article_texts(
    url: str,
    extracted_text: Optional[str],
    cleaned_html_text: Optional[str],
    embedding_blob: Optional[bytes]
) -> None:
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''
        UPDATE articles
        SET extracted_text = COALESCE(?, extracted_text),
            cleaned_html_text = COALESCE(?, cleaned_html_text),
            embedding = COALESCE(?, embedding)
        WHERE url = ?
        ''',
        (extracted_text, cleaned_html_text, embedding_blob, url)
    )
    conn.commit()
    conn.close()


# ##################################################################
# correct rank endpoint
# accepts user feedback and triggers retraining, always returns 200
@app.post("/correct_rank")
async def correct_rank(payload: CorrectRankRequest):
    try:
        exists, extracted_text, cleaned_html_text, embedding_blob = get_article_state(payload.url)
        predicted_score = None

        if not exists:
            extracted_text, cleaned_html_text = await fetch_article_texts(payload.url)
            if not extracted_text and not cleaned_html_text:
                return {"status": "skipped", "message": "Could not extract text from URL."}
            text_for_embedding = extracted_text or cleaned_html_text
            if text_for_embedding:
                predicted_score = engine.predict_score(text_for_embedding)
            embedding_blob = engine.get_embedding_bytes(text_for_embedding) if text_for_embedding else None
            store_new_article_with_target(
                payload.url,
                extracted_text,
                cleaned_html_text,
                embedding_blob,
                payload.score
            )
        else:
            text_for_prediction = extracted_text or cleaned_html_text
            if text_for_prediction:
                predicted_score = engine.predict_score(text_for_prediction)
            update_article_target(payload.url, payload.score)
            needs_fetch = extracted_text is None or cleaned_html_text is None
            if needs_fetch:
                fetched_extracted, fetched_cleaned = await fetch_article_texts(payload.url)
                updated_extracted = fetched_extracted if extracted_text is None else None
                updated_cleaned = fetched_cleaned if cleaned_html_text is None else None
                updated_embedding = None
                if embedding_blob is None:
                    text_for_embedding = updated_extracted or extracted_text or updated_cleaned or cleaned_html_text
                    if text_for_embedding:
                        updated_embedding = engine.get_embedding_bytes(text_for_embedding)

                if updated_extracted or updated_cleaned or updated_embedding:
                    update_article_texts(
                        payload.url,
                        updated_extracted,
                        updated_cleaned,
                        updated_embedding
                    )

        if predicted_score is not None:
            record_correction(payload.url, predicted_score, payload.score)
            delta = payload.score - predicted_score
            training_log(f"Correction: predicted={predicted_score:.2f}, actual={payload.score:.1f}, delta={delta:+.2f}")

        if training_loop:
            training_loop.add_epochs(CORRECTION_EPOCHS)
        return {"status": "accepted", "message": "Feedback received. Training epochs added."}
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
async def train_bulk(payload: TrainBulkRequest):
    try:
        count = 0
        for item in payload.items:
            url = item.get("url")
            score = item.get("score")
            if not url or score is None:
                continue
            set_training_target(url, float(score))
            count += 1

        if count > 0 and training_loop:
            training_loop.add_epochs(CORRECTION_EPOCHS)

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
            "SELECT url, extracted_text, cleaned_html_text, embedding FROM articles "
            "WHERE user_target_rank IS NOT NULL AND (extracted_text IS NULL OR cleaned_html_text IS NULL OR embedding IS NULL)"
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return {"status": "ok", "fetched": 0, "message": "All training URLs already have text."}

        fetched = 0
        failed = 0
        for url, extracted_text, cleaned_html_text, embedding in rows:
            updated_extracted = None
            updated_cleaned = None

            if extracted_text is None or cleaned_html_text is None:
                print(f"Fetching: {url[:60]}...")
                fetched_extracted, fetched_cleaned = await fetch_article_texts(url)
                updated_extracted = fetched_extracted if extracted_text is None else None
                updated_cleaned = fetched_cleaned if cleaned_html_text is None else None

            updated_embedding = None
            if embedding is None:
                text_for_embedding = updated_extracted or extracted_text or updated_cleaned or cleaned_html_text
                if text_for_embedding:
                    updated_embedding = engine.get_embedding_bytes(text_for_embedding)

            updated_any = bool(updated_extracted or updated_cleaned or updated_embedding)
            if updated_any:
                update_article_texts(url, updated_extracted, updated_cleaned, updated_embedding)
                fetched += 1
                stored_length = len(
                    updated_cleaned or updated_extracted or cleaned_html_text or extracted_text or ""
                )
                print(f"  OK ({stored_length} chars)")
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
# adds training epochs to the queue
@app.post("/retrain")
async def retrain_endpoint():
    try:
        stats = get_training_stats()
        if stats["trained_articles"] == 0:
            return {"status": "error", "message": "No training data available."}

        if training_loop:
            training_loop.add_epochs(RETRAIN_EPOCHS)
        return {
            "status": "accepted",
            "training_samples": stats["trained_articles"],
            "message": f"Added {RETRAIN_EPOCHS} training epochs."
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ##################################################################
# stats endpoint
# returns training statistics including accuracy metrics
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

        cursor.execute("SELECT COUNT(*) FROM corrections")
        total_corrections = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(delta), AVG(ABS(delta)) FROM corrections")
        correction_row = cursor.fetchone()
        avg_delta = correction_row[0]
        avg_abs_delta = correction_row[1]

        cursor.execute("""
            SELECT AVG(delta), AVG(ABS(delta)), COUNT(*)
            FROM corrections
            WHERE created_at >= datetime('now', '-7 days')
        """)
        recent_row = cursor.fetchone()
        recent_avg_delta = recent_row[0]
        recent_avg_abs_delta = recent_row[1]
        recent_count = recent_row[2]

        cursor.execute("""
            SELECT id, started_at, completed_at, training_type, samples_count,
                   epochs_run, final_loss, best_loss, status
            FROM training_history
            ORDER BY id DESC
            LIMIT 1
        """)
        last_training_row = cursor.fetchone()
        last_training = None
        if last_training_row:
            last_training = {
                "started_at": last_training_row[1],
                "completed_at": last_training_row[2],
                "type": last_training_row[3],
                "samples": last_training_row[4],
                "epochs": last_training_row[5],
                "final_loss": round(last_training_row[6], 4) if last_training_row[6] else None,
                "best_loss": round(last_training_row[7], 4) if last_training_row[7] else None,
                "status": last_training_row[8]
            }

        cursor.execute("SELECT COUNT(*) FROM training_history WHERE status = 'completed'")
        total_trainings = cursor.fetchone()[0]

        conn.close()

        return {
            "training_samples": trained,
            "with_embeddings": with_embeddings,
            "avg_target_score": round(avg_target, 2) if avg_target else None,
            "high_quality_count": high_quality,
            "low_quality_count": low_quality,
            "accuracy": {
                "total_corrections": total_corrections,
                "avg_delta": round(avg_delta, 2) if avg_delta else None,
                "avg_abs_delta": round(avg_abs_delta, 2) if avg_abs_delta else None,
                "recent_7d": {
                    "count": recent_count,
                    "avg_delta": round(recent_avg_delta, 2) if recent_avg_delta else None,
                    "avg_abs_delta": round(recent_avg_abs_delta, 2) if recent_avg_abs_delta else None
                }
            },
            "last_training": last_training,
            "total_trainings": total_trainings
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ##################################################################
# training set endpoint
# returns all trained URLs and their user target scores
@app.get("/training_set")
async def training_set_endpoint():
    try:
        data = training_set()
        return {"items": data, "count": len(data)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ##################################################################
# main
# entry point that starts the uvicorn server
def main() -> None:
    setproctitle.setproctitle("news-ranker-server")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, workers=1)


# ##################################################################
# crash protection constants
MAX_RESTART_ATTEMPTS = 10
RESTART_DELAY_SECONDS = 5
RESTART_BACKOFF_MULTIPLIER = 2
MAX_RESTART_DELAY_SECONDS = 300


# ##################################################################
# run with crash protection
# wraps main() to catch all exceptions and restart automatically
def run_with_crash_protection() -> None:
    import traceback
    from datetime import datetime

    attempt = 0
    delay = RESTART_DELAY_SECONDS

    while attempt < MAX_RESTART_ATTEMPTS:
        try:
            attempt += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if attempt > 1:
                print(f"[{timestamp}] Restart attempt {attempt}/{MAX_RESTART_ATTEMPTS}")
            main()
            # If main() returns normally (clean shutdown), don't restart
            print(f"[{timestamp}] Server stopped normally")
            break
        except KeyboardInterrupt:
            print("\nShutdown requested by user")
            break
        except SystemExit as e:
            # Respect explicit exit requests
            if e.code == 0:
                print("Server exited cleanly")
            else:
                print(f"Server exited with code {e.code}")
            break
        except Exception as e:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{timestamp}] CRASH DETECTED: {type(e).__name__}: {e}")
            print(f"Stack trace:\n{traceback.format_exc()}")

            if attempt < MAX_RESTART_ATTEMPTS:
                print(f"[{timestamp}] Restarting in {delay} seconds...")
                time.sleep(delay)
                # Exponential backoff
                delay = min(delay * RESTART_BACKOFF_MULTIPLIER, MAX_RESTART_DELAY_SECONDS)
            else:
                print(f"[{timestamp}] Max restart attempts ({MAX_RESTART_ATTEMPTS}) reached. Giving up.")
                raise


# ##################################################################
# entry point
# standard python pattern for running as script
if __name__ == "__main__":
    run_with_crash_protection()
