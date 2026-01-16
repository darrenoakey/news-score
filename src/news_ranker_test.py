import os
import pytest

from src.news_ranker import (
    MIN_SCORE,
    MAX_SCORE,
    EMBEDDING_DIM,
    PROJECT_ROOT,
    initialize_database,
    get_database_connection,
    ScoringEngine,
    RankingHead,
    HierarchicalAttentionRanker,
    compute_calibration_params,
)


# ##################################################################
# test paths
# use output/testing directory relative to project root
TEST_OUTPUT_DIR = PROJECT_ROOT / "output" / "testing"
TEST_DB_FILE = str(TEST_OUTPUT_DIR / "test_news_ranker.db")


# ##################################################################
# setup test database
# ensures clean database state before each test
@pytest.fixture(autouse=True)
def setup_test_database(monkeypatch):
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if os.path.exists(TEST_DB_FILE):
        os.remove(TEST_DB_FILE)
    monkeypatch.setattr("src.news_ranker.DB_FILE", TEST_DB_FILE)
    initialize_database()
    yield
    if os.path.exists(TEST_DB_FILE):
        os.remove(TEST_DB_FILE)


# ##################################################################
# test initialize database creates table
# verifies the articles table is created with correct schema
def test_initialize_database_creates_table():
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='articles'")
    result = cursor.fetchone()
    conn.close()
    assert result is not None
    assert result[0] == "articles"


# ##################################################################
# test initialize database adds cleaned html column
# verifies the articles table includes cleaned_html_text column
def test_initialize_database_adds_cleaned_html_column():
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(articles)")
    columns = {row[1] for row in cursor.fetchall()}
    conn.close()
    assert "cleaned_html_text" in columns


# ##################################################################
# test ranking head forward
# verifies the neural network produces correct output shape
def test_ranking_head_forward():
    import torch
    ranker = RankingHead(EMBEDDING_DIM)
    test_input = torch.randn(1, EMBEDDING_DIM)
    output = ranker(test_input)
    assert output.shape == (1, 1)


# ##################################################################
# test hierarchical attention ranker forward
# verifies hierarchical attention produces a scalar output
def test_hierarchical_attention_ranker_forward():
    import torch
    ranker = HierarchicalAttentionRanker(EMBEDDING_DIM)
    chunk = torch.randn(3, EMBEDDING_DIM)
    output = ranker([chunk])
    assert output.shape == (1,)


# ##################################################################
# test scoring engine initialization
# verifies engine loads models without error
def test_scoring_engine_initialization():
    engine = ScoringEngine()
    assert engine.embedder is not None
    assert engine.ranker is not None
    assert engine.is_training is False


# ##################################################################
# test split into sentences empty
# verifies empty text returns empty sentence list
def test_split_into_sentences_empty():
    engine = ScoringEngine()
    sentences = engine.split_into_sentences("")
    assert sentences == []


# ##################################################################
# test split into sentences
# verifies sentence splitting respects punctuation
def test_split_into_sentences():
    engine = ScoringEngine()
    text = "First sentence. Second sentence! Third sentence?"
    sentences = engine.split_into_sentences(text)
    assert sentences == ["First sentence.", "Second sentence!", "Third sentence?"]


# ##################################################################
# test chunk sentences short
# verifies short sentence list returns single chunk
def test_chunk_sentences_short():
    engine = ScoringEngine()
    sentences = ["Short sentence one.", "Short sentence two."]
    chunks = engine.chunk_sentences(sentences)
    assert len(chunks) == 1
    assert chunks[0] == sentences


# ##################################################################
# test chunk sentences long
# verifies long sentence list is split into multiple chunks
def test_chunk_sentences_long():
    engine = ScoringEngine()
    sentences = ["word " * 250, "word " * 250, "word " * 250]
    chunks = engine.chunk_sentences(sentences)
    assert len(chunks) == 2


# ##################################################################
# test build hierarchical embeddings produces correct dimension
# verifies embeddings return sentence vectors with expected size
def test_build_hierarchical_embeddings_produces_correct_dimension():
    engine = ScoringEngine()
    text = "This is a test article about technology and programming."
    chunks = engine.build_hierarchical_embeddings(text)
    assert len(chunks) == 1
    assert chunks[0].shape[1] == EMBEDDING_DIM


# ##################################################################
# test build hierarchical embeddings empty returns empty
# verifies empty text returns no chunks
def test_build_hierarchical_embeddings_empty_returns_empty():
    engine = ScoringEngine()
    chunks = engine.build_hierarchical_embeddings("")
    assert chunks == []


# ##################################################################
# test compute document mean vector
# verifies document mean vector is computed from chunks
def test_compute_document_mean_vector():
    import torch
    engine = ScoringEngine()
    chunk = torch.ones(3, EMBEDDING_DIM)
    vector = engine.compute_document_mean_vector([chunk])
    assert vector is not None
    assert vector.shape == (EMBEDDING_DIM,)


# ##################################################################
# test predict score returns valid range
# verifies scores are clamped between min and max
def test_predict_score_returns_valid_range():
    engine = ScoringEngine()
    text = "Deep technical analysis of machine learning algorithms and neural networks."
    score = engine.predict_score(text)
    assert MIN_SCORE <= score <= MAX_SCORE


# ##################################################################
# test predict score long text
# verifies long articles can be scored without error
def test_predict_score_long_text():
    engine = ScoringEngine()
    text = "Technical analysis of software engineering practices. " * 300
    score = engine.predict_score(text)
    assert MIN_SCORE <= score <= MAX_SCORE


# ##################################################################
# test get embedding bytes
# verifies embedding can be serialized and deserialized
def test_get_embedding_bytes():
    import pickle
    engine = ScoringEngine()
    text = "Test article content."
    embedding_bytes = engine.get_embedding_bytes(text)
    assert isinstance(embedding_bytes, bytes)
    chunks = pickle.loads(embedding_bytes)
    assert isinstance(chunks, list)
    assert chunks
    assert chunks[0].shape[1] == EMBEDDING_DIM


# ##################################################################
# test clamp score low
# verifies scores below minimum are clamped up
def test_clamp_score_low():
    engine = ScoringEngine()
    assert engine.clamp_score(-5.0) == MIN_SCORE
    assert engine.clamp_score(0.0) == MIN_SCORE


# ##################################################################
# test clamp score high
# verifies scores above maximum are clamped down
def test_clamp_score_high():
    engine = ScoringEngine()
    assert engine.clamp_score(15.0) == MAX_SCORE
    assert engine.clamp_score(100.0) == MAX_SCORE


# ##################################################################
# test clamp score valid
# verifies valid scores pass through unchanged
def test_clamp_score_valid():
    engine = ScoringEngine()
    assert engine.clamp_score(5.5) == 5.5
    assert engine.clamp_score(MIN_SCORE) == MIN_SCORE
    assert engine.clamp_score(MAX_SCORE) == MAX_SCORE


# ##################################################################
# test retrain with training data
# verifies model can learn from user feedback
def test_retrain_with_training_data():
    engine = ScoringEngine()

    # create training data: tech article should score high, gossip low
    tech_text = "Deep technical analysis of Python async IO programming. " * 100
    gossip_text = "Celebrity rumors and clickbait content. " * 10

    # store articles with embeddings and user targets
    conn = get_database_connection()
    tech_embedding = engine.get_embedding_bytes(tech_text)
    gossip_embedding = engine.get_embedding_bytes(gossip_text)

    conn.execute(
        "INSERT INTO articles (url, extracted_text, embedding, current_rank, user_target_rank) VALUES (?, ?, ?, ?, ?)",
        ("https://example.com/tech", tech_text, tech_embedding, 5.0, 10.0)
    )
    conn.execute(
        "INSERT INTO articles (url, extracted_text, embedding, current_rank, user_target_rank) VALUES (?, ?, ?, ?, ?)",
        ("https://example.com/gossip", gossip_text, gossip_embedding, 5.0, 1.0)
    )
    conn.commit()
    conn.close()

    # retrain
    engine.retrain()

    # verify model learned the distinction
    tech_score = engine.predict_score(tech_text)
    gossip_score = engine.predict_score(gossip_text)

    assert tech_score > gossip_score
    assert tech_score > 6.0


# ##################################################################
# test retrain no data
# verifies retrain handles empty training set gracefully
def test_retrain_no_data():
    engine = ScoringEngine()
    engine.retrain()  # should not raise
    assert engine.is_training is False


# ##################################################################
# test clean html text
# verifies html tags are stripped and whitespace normalized
def test_clean_html_text():
    from src.news_ranker import clean_html_text
    html = "<p>Hello   <b>world</b></p>  <div>test</div>"
    result = clean_html_text(html)
    assert result == "Hello world test"


# ##################################################################
# test clean html text empty
# verifies empty input returns empty string
def test_clean_html_text_empty():
    from src.news_ranker import clean_html_text
    assert clean_html_text("") == ""
    assert clean_html_text("   ") == ""


# ##################################################################
# test compute calibration params
# verifies min/max mapping for calibration
def test_compute_calibration_params():
    scale, bias = compute_calibration_params(2.0, 4.0, 1.0, 10.0)
    assert abs((2.0 * scale + bias) - 1.0) < 1e-6
    assert abs((4.0 * scale + bias) - 10.0) < 1e-6


# ##################################################################
# test compute calibration params no range
# verifies zero-range predictions keep defaults
def test_compute_calibration_params_no_range():
    scale, bias = compute_calibration_params(2.0, 2.0, 1.0, 10.0)
    assert scale == 1.0
    assert bias == 0.0


# ##################################################################
# test fetch with trafilatura real url
# verifies trafilatura can extract text from a real static page
def test_fetch_with_trafilatura_real_url():
    from src.news_ranker import fetch_with_trafilatura
    text = fetch_with_trafilatura("https://example.com")
    assert text is not None
    assert len(text) > 50


# ##################################################################
# test fetch with trafilatura invalid url
# verifies invalid urls return none without error
def test_fetch_with_trafilatura_invalid_url():
    from src.news_ranker import fetch_with_trafilatura
    text = fetch_with_trafilatura("https://this-domain-does-not-exist-12345.com")
    assert text is None


# ##################################################################
# test fetch with playwright real url
# verifies playwright can extract text from a real page
def test_fetch_with_playwright_real_url():
    from src.news_ranker import fetch_with_playwright
    text = fetch_with_playwright("https://example.com")
    assert text is not None
    assert len(text) > 50
