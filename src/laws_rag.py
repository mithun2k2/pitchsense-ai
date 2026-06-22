"""
laws_rag.py
Docling-powered RAG pipeline for FIFA Laws of the Game.

Pipeline:
  1. Download the FIFA Laws PDF (or use a local copy)
  2. Docling parses it into structured chunks
  3. Chunks are embedded with sentence-transformers (all-MiniLM-L6-v2)
  4. On query, retrieve the top-k most relevant chunks
  5. Return chunks as context for Granite's VAR explanations

No vector database required — we use simple cosine similarity in-memory,
keeping the dependency footprint minimal for a hackathon prototype.
"""

import os
import json
import hashlib
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Optional imports — graceful degradation if not installed
# ---------------------------------------------------------------------------

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = False
except ImportError:
    DOCLING_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LAWS_PDF_URL = (
    "https://digitalhub.fifa.com/m/5371a6daa42948af/"
    "original/Laws-of-the-Game-2024-25-EN.pdf"
)

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
LAWS_PDF_PATH = DATA_DIR / "fifa_laws.pdf"
CHUNKS_CACHE = MODELS_DIR / "laws_chunks.pkl"
EMBEDDINGS_CACHE = MODELS_DIR / "laws_embeddings.npy"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4
CHUNK_MIN_CHARS = 120   # discard headers / page numbers shorter than this


# ---------------------------------------------------------------------------
# VAR-relevant Law numbers for keyword boosting
# ---------------------------------------------------------------------------

VAR_RELEVANT_LAWS = {
    "Law 12",  # Fouls and misconduct
    "Law 11",  # Offside
    "Law 10",  # Determining the outcome
    "Law 13",  # Free kicks
    "Law 14",  # Penalty kick
    "Law 3",   # Players (e.g. substitutions triggering VAR)
}


# ---------------------------------------------------------------------------
# Step 1 — Download the Laws PDF
# ---------------------------------------------------------------------------

def download_laws_pdf(force: bool = False) -> Path:
    """Download the FIFA Laws PDF if not already cached."""
    DATA_DIR.mkdir(exist_ok=True)
    if LAWS_PDF_PATH.exists() and not force:
        print(f"[laws_rag] Using cached PDF: {LAWS_PDF_PATH}")
        return LAWS_PDF_PATH

    print(f"[laws_rag] Downloading FIFA Laws PDF from {LAWS_PDF_URL} ...")
    resp = requests.get(LAWS_PDF_URL, timeout=60)
    resp.raise_for_status()
    LAWS_PDF_PATH.write_bytes(resp.content)
    size_kb = LAWS_PDF_PATH.stat().st_size / 1024
    print(f"[laws_rag] Saved {size_kb:.1f} KB → {LAWS_PDF_PATH}")
    return LAWS_PDF_PATH


# ---------------------------------------------------------------------------
# Step 2 — Parse with Docling (or fallback to basic text extraction)
# ---------------------------------------------------------------------------

def _parse_with_docling(pdf_path: Path) -> list[dict]:
    """Use Docling to convert the PDF into structured text chunks."""
    print("[laws_rag] Parsing with Docling ...")
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    chunks = []
    current_law = "Preamble"

    for element in doc.texts:
        text = element.text.strip()
        if not text or len(text) < CHUNK_MIN_CHARS:
            continue

        # Track which Law we're in for metadata
        if text.upper().startswith("LAW ") and len(text) < 60:
            current_law = text.title()
            continue

        chunks.append({
            "text": text,
            "law": current_law,
            "source": "FIFA Laws of the Game 2024/25",
        })

    print(f"[laws_rag] Docling extracted {len(chunks)} chunks.")
    return chunks


def _parse_fallback(pdf_path: Path) -> list[dict]:
    """
    Fallback parser using PyMuPDF (fitz) when Docling is not available.
    Groups text by page, then splits on double newlines.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError(
            "Neither Docling nor PyMuPDF (fitz) is available. "
            "Install one: pip install docling   OR   pip install pymupdf"
        )

    print("[laws_rag] Parsing with PyMuPDF fallback ...")
    doc = fitz.open(str(pdf_path))
    chunks = []
    current_law = "Preamble"

    for page in doc:
        text = page.get_text()
        for para in text.split("\n\n"):
            para = para.strip()
            if not para or len(para) < CHUNK_MIN_CHARS:
                continue
            if para.upper().startswith("LAW ") and len(para) < 60:
                current_law = para.title()
                continue
            chunks.append({
                "text": para,
                "law": current_law,
                "source": "FIFA Laws of the Game 2024/25",
            })

    print(f"[laws_rag] PyMuPDF extracted {len(chunks)} chunks.")
    return chunks


def parse_laws_pdf(pdf_path: Path) -> list[dict]:
    """Parse the Laws PDF, preferring Docling, falling back to PyMuPDF."""
    if DOCLING_AVAILABLE:
        return _parse_with_docling(pdf_path)
    else:
        print("[laws_rag] Docling not installed — using PyMuPDF fallback.")
        return _parse_fallback(pdf_path)


# ---------------------------------------------------------------------------
# Step 3 — Embed chunks
# ---------------------------------------------------------------------------

def _load_embed_model() -> "SentenceTransformer":
    if not ST_AVAILABLE:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        )
    print(f"[laws_rag] Loading embedding model: {EMBED_MODEL_NAME}")
    return SentenceTransformer(EMBED_MODEL_NAME)


def embed_chunks(chunks: list[dict]) -> np.ndarray:
    """Embed all chunk texts and return a (N, D) float32 array."""
    model = _load_embed_model()
    texts = [c["text"] for c in chunks]
    print(f"[laws_rag] Embedding {len(texts)} chunks ...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 4 — Build and cache the index
# ---------------------------------------------------------------------------

def build_index(force: bool = False) -> tuple[list[dict], np.ndarray]:
    """
    Full pipeline: download → parse → embed → cache.
    Returns (chunks, embeddings).
    """
    MODELS_DIR.mkdir(exist_ok=True)

    # Return cached if available and not forced
    if CHUNKS_CACHE.exists() and EMBEDDINGS_CACHE.exists() and not force:
        print("[laws_rag] Loading cached index ...")
        with open(CHUNKS_CACHE, "rb") as f:
            chunks = pickle.load(f)
        embeddings = np.load(EMBEDDINGS_CACHE)
        print(f"[laws_rag] Loaded {len(chunks)} chunks from cache.")
        return chunks, embeddings

    pdf_path = download_laws_pdf()
    chunks = parse_laws_pdf(pdf_path)

    embeddings = embed_chunks(chunks)

    # Cache to disk
    with open(CHUNKS_CACHE, "wb") as f:
        pickle.dump(chunks, f)
    np.save(EMBEDDINGS_CACHE, embeddings)
    print(f"[laws_rag] Index cached: {len(chunks)} chunks.")

    return chunks, embeddings


# ---------------------------------------------------------------------------
# Step 5 — Retrieval
# ---------------------------------------------------------------------------

class LawsRAG:
    """
    In-memory retrieval over FIFA Laws chunks.
    Loaded once at startup via build_index().
    """

    def __init__(self):
        self._chunks: Optional[list[dict]] = None
        self._embeddings: Optional[np.ndarray] = None
        self._embed_model = None
        self._ready = False

    def load(self, force_rebuild: bool = False):
        """Load or build the index. Call once at app startup."""
        try:
            self._chunks, self._embeddings = build_index(force=force_rebuild)
            self._embed_model = _load_embed_model()
            self._ready = True
            print("[laws_rag] RAG system ready.")
        except Exception as e:
            print(f"[laws_rag] WARNING: RAG unavailable — {e}")
            self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Retrieve top_k most relevant Law chunks for a query.

        Returns list of dicts: {"text", "law", "source", "score"}
        """
        if not self._ready:
            return []

        # Embed query
        q_vec = self._embed_model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)[0]

        # Cosine similarity (embeddings already normalised → dot product)
        scores = self._embeddings @ q_vec

        # Boost VAR-relevant laws
        for i, chunk in enumerate(self._chunks):
            for law in VAR_RELEVANT_LAWS:
                if law in chunk.get("law", ""):
                    scores[i] *= 1.15
                    break

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                **self._chunks[idx],
                "score": float(scores[idx]),
            })
        return results

    def retrieve_as_context(self, query: str, top_k: int = TOP_K) -> str:
        """
        Retrieve and format chunks as a single context string
        ready to inject into a Granite prompt.
        """
        chunks = self.retrieve(query, top_k=top_k)
        if not chunks:
            return ""

        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(
                f"[{i}] {c['law']} — {c['source']}\n{c['text']}"
            )
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Singleton for use in Streamlit (cached via st.cache_resource)
# ---------------------------------------------------------------------------

_rag_singleton: Optional[LawsRAG] = None


def get_rag() -> LawsRAG:
    """Return the module-level RAG singleton, initialising on first call."""
    global _rag_singleton
    if _rag_singleton is None:
        _rag_singleton = LawsRAG()
        _rag_singleton.load()
    return _rag_singleton


# ---------------------------------------------------------------------------
# CLI entry point — run this file directly to build the index
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build FIFA Laws RAG index")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if cache exists")
    parser.add_argument("--query", type=str, help="Test retrieval with a query after building")
    args = parser.parse_args()

    rag = LawsRAG()
    rag.load(force_rebuild=args.force)

    if args.query:
        print(f"\n--- Query: {args.query!r} ---")
        print(rag.retrieve_as_context(args.query))
