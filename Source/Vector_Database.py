"""
Vector Database Pipeline
=========================
Consumes scraped markdown files from Data_Scraper.py and manages a
ChromaDB vector store for semantic search using LangChain.

Pipeline:
    1. Scan OUTPUT_DIR for scraped markdown files
    2. Read metadata (URL, hash, category) from file headers
    3. Compare hashes against existing vectors to detect changes
    4. Chunk changed/new files and embed into ChromaDB
    5. Purge vectors for files that no longer exist
    6. Query the database with semantic similarity search

Usage:
    from Vector_Database import VectorDatabase

    vdb = VectorDatabase()
    vdb.process()
    results = vdb.query("How do I use CPF for housing?")

Dependencies:
    pip install langchain-text-splitters langchain-chroma langchain-openai pandas tqdm
"""

import os
from langchain_text_splitters import MarkdownTextSplitter
from langchain_chroma import Chroma
from tqdm import tqdm

from Source.Constants import Constants


class VectorDatabase:
    """
    Manages the embedding pipeline and vector store for scraped CPF content.

    Uses LangChain's MarkdownTextSplitter for structure-aware chunking
    and Chroma as the vector store with OpenAI embeddings.

    Source of truth: the scraped .md files in OUTPUT_DIR.
    """

    def __init__(self):
        """Initialise embedding model, splitter, and Chroma vector store."""
        self._print("Initialising Vector Database...")
        self.embeddings = self._init_embeddings()
        self.splitter = MarkdownTextSplitter(
            chunk_size=Constants.CHUNK_SIZE,
            chunk_overlap=Constants.CHUNK_OVERLAP,
        )
        self.vectorstore = Chroma(
            collection_name=Constants.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=Constants.PERSIST_DIR,
            collection_metadata={"hnsw:space": Constants.DISTANCE_METRIC},
        )
        self._print(f"  Collection '{Constants.COLLECTION_NAME}' ready")

    def _init_embeddings(self):
        """Initialise embedding model based on configuration flag."""
        if Constants.USE_LOCAL_EMBEDDINGS:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            from langchain_core.embeddings import Embeddings

            class LocalEmbeddings(Embeddings):
                """LangChain-compatible wrapper around ChromaDB's default ONNX embeddings."""
                def __init__(self):
                    self._fn = DefaultEmbeddingFunction()

                def embed_documents(self, texts):
                    return self._fn(texts)

                def embed_query(self, text):
                    return self._fn([text])[0]

            self._print("  Using local embeddings (ChromaDB default)")
            return LocalEmbeddings()
        else:
            from langchain_openai import OpenAIEmbeddings
            self._print("  Using OpenAI embeddings")
            return OpenAIEmbeddings(model=Constants.EMBEDDING_MODEL)

    def _print(self, msg):
        """Print a message only when VERBOSE mode is enabled."""
        if Constants.VERBOSE:
            print(msg)

    # ==========================================================================
    # FILE SCANNING
    # ==========================================================================

    def _scan_files(self):
        """
        Scan OUTPUT_DIR for all scraped .md files and parse their headers.

        Returns:
            list[dict]: Each dict contains 'filepath', 'url', 'hash', 'category'.
        """
        files = []
        for root, _, filenames in os.walk(Constants.OUTPUT_DIR):
            for fname in filenames:
                if fname.endswith(".md"):
                    filepath = os.path.join(root, fname)
                    metadata = self._parse_header(filepath)
                    if metadata:
                        metadata["filepath"] = filepath
                        files.append(metadata)
        return files

    def _parse_header(self, filepath):
        """
        Read the first FILE_HEADER_LINES of a file and extract metadata.

        Expected format:
            URL: https://...
            Hash: abc123...
            Category: account services/fighting scams
            Date Scraped: 20250715
            ====...

        Returns:
            dict: With keys 'url', 'hash', 'category', or None if malformed.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = [f.readline() for _ in range(Constants.FILE_HEADER_LINES)]

            url = lines[0].strip().removeprefix("URL: ")
            content_hash = lines[1].strip().removeprefix("Hash: ")
            category = lines[2].strip().removeprefix("Category: ")

            return {"url": url, "hash": content_hash, "category": category}
        except (IndexError, IOError):
            return None

    def _load_content(self, filepath):
        """
        Load markdown content from file, skipping the header.

        Returns:
            str: Markdown content below the separator.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return "".join(lines[Constants.FILE_HEADER_LINES:])

    # ==========================================================================
    # CHUNKING
    # ==========================================================================

    def _chunk_text(self, text):
        """
        Split markdown text into structure-aware chunks.

        Uses LangChain's MarkdownTextSplitter which splits by:
            1. Headings (##, ###)
            2. Paragraphs (double newline)
            3. Lines (single newline)
            4. Words (space) — last resort

        Returns:
            list[str]: List of text chunks.
        """
        if not text.strip():
            return []
        docs = self.splitter.create_documents([text])
        return [doc.page_content for doc in docs]

    # ==========================================================================
    # VECTOR OPERATIONS
    # ==========================================================================

    def _get_existing_hashes(self):
        """
        Retrieve all unique (url, content_hash) pairs currently in the vector store.

        Returns:
            dict: Mapping of URL → content_hash for all stored vectors.
        """
        collection = self.vectorstore._collection
        all_data = collection.get(include=["metadatas"])
        url_hashes = {}
        for meta in all_data["metadatas"]:
            url_hashes[meta["url"]] = meta.get("content_hash", "")
        return url_hashes

    def _delete_vectors_for_url(self, url):
        """Remove all vectors associated with a given URL."""
        collection = self.vectorstore._collection
        existing = collection.get(where={"url": url})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

    def _embed_and_upsert(self, url, text, category, content_hash):
        """
        Chunk text, generate embeddings, and upsert into ChromaDB.

        Args:
            url: Source page URL
            text: Full scraped markdown content
            category: Category string from file header
            content_hash: SHA-256 hash from file header
        """
        chunks = self._chunk_text(text)
        if not chunks:
            return

        metadatas = [
            {
                "url": url,
                "category": category,
                "chunk_index": i,
                "content_hash": content_hash,
            }
            for i in range(len(chunks))
        ]
        ids = [f"{content_hash}_{i}" for i in range(len(chunks))]

        self.vectorstore.add_texts(
            texts=chunks,
            metadatas=metadatas,
            ids=ids,
        )

    # ==========================================================================
    # MAIN PROCESS
    # ==========================================================================

    def process(self, progress_callback=None):
        """
        Execute the full embedding pipeline.

        Args:
            progress_callback: Optional callable(current, total) for progress updates.

        Steps:
            1. Scan files and read their header metadata
            2. Compare file hashes against stored vector hashes
            3. Embed new/changed files, skip unchanged ones
            4. Purge vectors for files that no longer exist
        """
        # Scan current files
        files = self._scan_files()
        file_urls = {f["url"] for f in files}
        self._print(f"  Files found: {len(files)}")

        # Get existing state from vector store
        existing_hashes = self._get_existing_hashes()

        # Determine what needs updating
        to_embed = [f for f in files if f["hash"] != existing_hashes.get(f["url"])]
        to_purge = [url for url in existing_hashes if url not in file_urls]

        # Purge vectors for deleted files
        if to_purge:
            self._print(f"  Purging vectors for {len(to_purge)} removed files")
            for url in to_purge:
                self._delete_vectors_for_url(url)

        if not to_embed:
            self._print("  No pages need embedding. Database is up to date.")
            return {
                "files_found": len(files),
                "embedded": 0,
                "skipped": len(files),
                "purged": len(to_purge),
            }

        self._print(f"  Pages to embed: {len(to_embed)}")

        embedded_count = 0
        total_chunks = 0

        for i, file_info in enumerate(tqdm(to_embed, desc="  Embedding", disable=not Constants.VERBOSE)):
            if progress_callback:
                progress_callback(i + 1, len(to_embed))

            text = self._load_content(file_info["filepath"])

            if not text.strip():
                continue

            # Remove old vectors if this is an update
            if file_info["url"] in existing_hashes:
                self._delete_vectors_for_url(file_info["url"])

            self._embed_and_upsert(
                url=file_info["url"],
                text=text,
                category=file_info["category"],
                content_hash=file_info["hash"],
            )

            embedded_count += 1
            total_chunks += len(self._chunk_text(text))

        self._print(f"\n  Pages embedded  : {embedded_count}")
        self._print(f"  Total chunks    : {total_chunks}")
        self._print(f"  Vectors in DB   : {self.vectorstore._collection.count()}")

        return {
            "files_found": len(files),
            "embedded": embedded_count,
            "skipped": len(files) - len(to_embed),
            "purged": len(to_purge),
        }

    # ==========================================================================
    # QUERY
    # ==========================================================================

    def query(self, query_text, top_k=None, where_filter=None):
        """
        Perform semantic similarity search against the vector store.

        Args:
            query_text: Natural language query string.
            top_k: Number of most similar chunks to return (default from Constants).
            where_filter: Optional pre-built Chroma where filter dict.

        Returns:
            list[dict]: Ranked results with keys: text, url, category,
                        chunk_index, score.
        """
        if top_k is None:
            top_k = Constants.DEFAULT_TOP_K

        results = self.vectorstore.similarity_search_with_score(
            query=query_text,
            k=top_k,
            filter=where_filter,
        )

        formatted = []
        for doc, score in results:
            formatted.append({
                "text": doc.page_content,
                "url": doc.metadata.get("url", ""),
                "category": doc.metadata.get("category", ""),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "score": 1 - score,  # Convert distance to similarity
            })

        return formatted


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    vdb = VectorDatabase()
    vdb.process()
