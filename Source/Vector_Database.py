"""
Vector Database Pipeline
=========================
Consumes output from Data_Scraper.py (log file + scraped text files) and
manages a ChromaDB vector store for semantic search.

Pipeline:
    1. Read scrape_log.csv to identify pages needing embedding
    2. Load scraped text files and chunk them
    3. Generate embeddings and upsert into ChromaDB
    4. Remove stale vectors for pages whose content has changed
    5. Query the database with semantic similarity search

Usage:
    from Vector_Database import VectorDatabase

    vdb = VectorDatabase()
    vdb.process()  # Embed all pages flagged Needs_Embedding=True
    results = vdb.query("How do I use CPF for housing?")

Dependencies:
    pip install chromadb sentence-transformers pandas tqdm
"""

import os
import pandas as pd
from urllib.parse import urlparse
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

from Source.Constants import Constants


class VectorDatabase:
    """
    Manages the embedding pipeline and vector store for scraped CPF content.

    Workflow:
        1. Read the scrape log to determine which files need embedding
        2. Delete stale vectors for changed pages
        3. Chunk text files into overlapping segments
        4. Generate embeddings via SentenceTransformer
        5. Upsert chunks with metadata into ChromaDB
        6. Provide semantic search via query()
    """

    def __init__(self):
        """Initialise embedding model, ChromaDB client, and collection."""
        self._print("Initialising Vector Database...")
        self.model = SentenceTransformer(Constants.EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=Constants.PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name=Constants.COLLECTION_NAME,
            metadata={"hnsw:space": Constants.DISTANCE_METRIC},
        )
        self._print(f"  Collection '{Constants.COLLECTION_NAME}' ready ({self.collection.count()} vectors)")

    def _print(self, msg):
        """Print a message only when VERBOSE mode is enabled."""
        if Constants.VERBOSE:
            print(msg)

    # ==========================================================================
    # LOG PARSING
    # ==========================================================================

    def _get_pages_needing_embedding(self):
        """
        Read the scrape log and return rows where embedding is required.

        Returns:
            pd.DataFrame: Rows with Scraped=Y and Needs_Embedding=True.
        """
        if not os.path.exists(Constants.LOG_FILE):
            self._print("  No log file found. Run Data_Scraper first.")
            return pd.DataFrame()

        df = pd.read_csv(Constants.LOG_FILE)
        return df[
            (df[Constants.COL_SCRAPED] == Constants.SCRAPED_YES) &
            (df[Constants.COL_NEEDS_EMBEDDING] == True)
        ]

    # ==========================================================================
    # TEXT LOADING
    # ==========================================================================

    def _url_to_filepath(self, url):
        """
        Convert a URL to its local file path (mirrors Data_Scraper logic).

        Returns:
            str: Full path to the scraped text file.
        """
        parsed = urlparse(url)
        parts = [p for p in parsed.path.strip("/").split("/") if p]

        if not parts:
            return os.path.join(Constants.OUTPUT_DIR, "index.txt")

        folder = os.path.join(Constants.OUTPUT_DIR, *parts[:-1]) if len(parts) > 1 else Constants.OUTPUT_DIR
        filename = f"{parts[-1]}.txt"
        return os.path.join(folder, filename)

    def _load_text(self, filepath):
        """
        Load scraped text from file, skipping the URL header and separator.

        Returns:
            str: Clean text content, or empty string if file not found.
        """
        if not os.path.exists(filepath):
            return ""
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Skip header lines (URL line + separator)
        return "".join(lines[Constants.FILE_HEADER_LINES:])

    # ==========================================================================
    # CHUNKING
    # ==========================================================================

    def _chunk_text(self, text):
        """
        Split text into overlapping chunks of fixed character length.

        Uses CHUNK_SIZE and CHUNK_OVERLAP to create segments that maintain
        context across boundaries.

        Returns:
            list[str]: List of text chunks.
        """
        if not text.strip():
            return []

        chunks = []
        start = 0
        while start < len(text):
            end = start + Constants.CHUNK_SIZE
            chunks.append(text[start:end])
            start = end - Constants.CHUNK_OVERLAP

        return chunks

    # ==========================================================================
    # VECTOR OPERATIONS
    # ==========================================================================

    def _delete_stale_vectors(self, url):
        """
        Remove all existing vectors for a given URL.

        Called before re-embedding a page whose content has changed,
        ensuring no stale chunks remain in the collection.
        """
        existing = self.collection.get(where={"url": url})
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])

    def _embed_and_upsert(self, url, text, category, content_hash, scraped_date):
        """
        Chunk text, generate embeddings, and upsert into ChromaDB.

        Each chunk is stored with metadata for filtering and traceability.

        Args:
            url: Source page URL
            text: Full scraped text content
            category: Page category from ALLOWED_PATHS
            content_hash: SHA-256 hash of the text
            scraped_date: Date the page was scraped (YYYYMMDD)
        """
        chunks = self._chunk_text(text)
        if not chunks:
            return

        # Generate embeddings for all chunks in one batch
        embeddings = self.model.encode(chunks).tolist()

        # Prepare batch data for ChromaDB
        ids = [f"{content_hash}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "url": url,
                "category": category or "",
                "chunk_index": i,
                "content_hash": content_hash,
                "scraped_date": scraped_date,
            }
            for i in range(len(chunks))
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

    # ==========================================================================
    # MAIN PROCESS
    # ==========================================================================

    def process(self):
        """
        Execute the full embedding pipeline.

        Steps:
            1. Read log to find pages needing embedding
            2. For each page: delete stale vectors, load text, chunk, embed, upsert
            3. Report summary statistics
        """
        to_embed = self._get_pages_needing_embedding()

        if to_embed.empty:
            self._print("  No pages need embedding. Database is up to date.")
            return

        self._print(f"  Pages to embed: {len(to_embed)}")

        embedded_count = 0
        total_chunks = 0

        for _, row in tqdm(to_embed.iterrows(), total=len(to_embed), desc="  Embedding", disable=not Constants.VERBOSE):
            url = row[Constants.COL_URL]
            filepath = self._url_to_filepath(url)
            text = self._load_text(filepath)

            if not text.strip():
                self._print(f"  SKIPPED (empty): {filepath}")
                continue

            # Remove old vectors for this URL before re-embedding
            self._delete_stale_vectors(url)

            # Embed and store
            self._embed_and_upsert(
                url=url,
                text=text,
                category=row.get(Constants.COL_CATEGORY),
                content_hash=row[Constants.COL_CONTENT_HASH],
                scraped_date=str(row.get(Constants.COL_SCRAPED_DATE, "")),
            )

            embedded_count += 1
            total_chunks += len(self._chunk_text(text))

        self._print(f"\n  Pages embedded  : {embedded_count}")
        self._print(f"  Total chunks    : {total_chunks}")
        self._print(f"  Vectors in DB   : {self.collection.count()}")

    # ==========================================================================
    # QUERY
    # ==========================================================================

    def query(self, query_text, top_k=None, categories=None):
        """
        Perform semantic similarity search against the vector store.

        Args:
            query_text: Natural language query string.
            top_k: Number of most similar chunks to return (default from Constants).
            categories: Optional list of categories to filter on.
                        e.g. ["Housing", "Retirement"]

        Returns:
            list[dict]: Ranked results with keys: text, url, category,
                        chunk_index, score.
        """
        if top_k is None:
            top_k = Constants.DEFAULT_TOP_K

        query_embedding = self.model.encode([query_text]).tolist()

        # Build filter if categories specified
        where_filter = None
        if categories:
            where_filter = {"category": {"$in": categories}}

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Format results into a clean list
        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "text": results["documents"][0][i],
                "url": results["metadatas"][0][i]["url"],
                "category": results["metadatas"][0][i]["category"],
                "chunk_index": results["metadatas"][0][i]["chunk_index"],
                "score": 1 - results["distances"][0][i],  # Convert distance to similarity
            })

        return formatted


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    vdb = VectorDatabase()
    vdb.process()
