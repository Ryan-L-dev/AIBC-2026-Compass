"""
Constants
=========
Centralised configuration constants used across the scraping and
vector database pipeline. Prevents magic text/numbers scattered
throughout the codebase.

Usage:
    from Constants import Constants

    print(Constants.SITEMAP_URL)
    print(Constants.CHUNK_SIZE)
"""

import os


class Constants:
    """Single source of truth for all configuration values."""

    # ==========================================================================
    # TARGET WEBSITE
    # ==========================================================================

    SITEMAP_URL = "https://www.cpf.gov.sg/sitemap.xml"
    BASE_DOMAIN = "www.cpf.gov.sg"

    # ==========================================================================
    # OUTPUT
    # ==========================================================================

    OUTPUT_DIR = "scraped_pages"
    LOG_FILE = "scrape_log.csv"

    # ==========================================================================
    # SCRAPING BEHAVIOUR
    # ==========================================================================

    DELAY = 1
    VERBOSE = True
    MAX_REDIRECTS = 10
    REQUEST_TIMEOUT = 10

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    # ==========================================================================
    # URL FILTERING
    # ==========================================================================

    ALLOWED_PATHS = {
        "/member/retirement-income": "Retirement",
        "/member/cpf-overview": "Overview",
        "/member/home-ownership": "Housing",
        "/member/healthcare-financing": "Healthcare",
        "/member/growing-your-savings": "Savings",
        "/member/account-services": "Account Services",
    }

    # ==========================================================================
    # HTML NOISE REMOVAL
    # ==========================================================================

    NOISE_TAGS = ["script", "style", "nav", "footer", "header"]
    NOISE_CLASSES = ["anchor-tab-list", "stick-tabs", "social-sharing-bar"]


    # Known template/boilerplate lines to strip from scraped content.
    # Discovered via template detection runs and manually curated.
    NOISE_LINES = ["true", "false", "null", "undefined", ""]

    # Template detection: lines appearing in more than this fraction of files
    # are flagged as potential template noise and written to TEMPLATE_LINES_FILE.
    TEMPLATE_LINE_THRESHOLD = 0.5  # 50% of files
    TEMPLATE_LINE_MIN_LENGTH = 10  # Ignore trivially short lines
    TEMPLATE_LINES_FILE = os.path.join("Resources", "template_lines.txt")

    # ==========================================================================
    # FILE FORMAT
    # ==========================================================================

    # Number of header lines in scraped text files (URL + Hash + Category + Date Scraped + separator)
    FILE_HEADER_LINES = 5
    FILE_SEPARATOR = "=" * 80

    # ==========================================================================
    # VECTOR DATABASE
    # ==========================================================================

    COLLECTION_NAME = "cpf_knowledge_base"
    PERSIST_DIR = "vector_store"
    EMBEDDING_MODEL = "text-embedding-3-small"
    DISTANCE_METRIC = "cosine"

    # Chunking
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50

    # Query
    DEFAULT_TOP_K = 5

    # ==========================================================================
    # LOG COLUMNS
    # ==========================================================================

    COL_URL = "URL"
    COL_SCRAPED = "Scraped"
    COL_REASON = "Reason"
    COL_CATEGORY = "Category"
    COL_SCRAPED_DATE = "Scraped_Date"
    COL_CONTENT_HASH = "Content_Hash"
    COL_NEEDS_EMBEDDING = "Needs_Embedding"

    SCRAPED_YES = "Y"
    SCRAPED_NO = "N"

    # ==========================================================================
    # CHAT ENGINE
    # ==========================================================================

    # History management
    MAX_HISTORY_TURNS = 5
    CHAT_HISTORY_FILE = "chat_history.json"

    # Retrieval
    CONFIDENCE_THRESHOLD = 0.3

    # LLM (Amazon Bedrock)
    LLM_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
    LLM_MAX_TOKENS = 1024
    LLM_REGION = "us-east-1"

    SYSTEM_PROMPT = (
        "You are a helpful assistant answering questions about CPF (Central Provident Fund) policies in Singapore. "
        "Answer using ONLY the provided context. "
        "If the context does not contain enough information to answer, say so clearly. "
        "Cite source URLs when referencing specific information."
    )

    SUMMARY_PROMPT = (
        "Summarise this conversation in 2-3 sentences. "
        "Preserve key topics, specific details (numbers, dates, policy names), "
        "and any user preferences or constraints mentioned."
    )
    SUMMARY_MAX_TOKENS = 200

    # Response fallbacks
    NO_ANSWER_MESSAGE = "I don't have information on that topic based on the available CPF policies."
    ERROR_MESSAGE = "Sorry, I encountered an error generating a response. Please try again."
