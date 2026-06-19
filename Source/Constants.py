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

    OUTPUT_DIR = os.path.join("Data", "scraped_pages")
    LOG_FILE = os.path.join("Data", "resources", "scrape_log.csv")

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

    ALLOWED_PATHS = [
        "/member/retirement-income",
        "/member/cpf-overview",
        "/member/home-ownership",
        "/member/healthcare-financing",
        "/member/growing-your-savings",
        "/member/account-services",
    ]

    # ==========================================================================
    # HTML NOISE REMOVAL
    # ==========================================================================

    NOISE_TAGS = ["script", "style", "nav", "footer", "header"]
    NOISE_CLASSES = [
        "anchor-tab-list",      # Tab navigation bar
        "stick-tabs",           # Sticky tab navigation
        "social-sharing-bar",   # Social media sharing buttons
        "content-change-log",   # Temporary site-wide notice banners (e.g. Singtel outage)
        "related-articles",     # "Related Reads" article carousel
    ]


    # Known template/boilerplate lines to strip from scraped content.
    # Discovered via template detection runs and manually curated.
    NOISE_LINES = ["true", "false", "null", "undefined", ""]

    # Lines that mark the start of trailing boilerplate sections.
    # All content from these markers onwards is truncated.
    TRUNCATE_MARKERS = [
        "\nResources\n",              # External links section (standalone heading)
        "### Common questions",       # FAQ links to cpf.gov.sg/service/article pages
        "### Need more information?", # Links to external resources (HDB, MOH, etc.)
        "### Related Reads",          # Article previews (fallback if CSS class missed)
    ]

    # Template detection: lines appearing in more than this fraction of files
    # are flagged as potential template noise and written to TEMPLATE_LINES_FILE.
    TEMPLATE_LINE_THRESHOLD = 0.5  # 50% of files
    TEMPLATE_LINE_MIN_LENGTH = 10  # Ignore trivially short lines
    TEMPLATE_LINES_FILE = os.path.join("Data", "resources", "template_lines.txt")
    CATEGORIES_LIST_FILE = os.path.join("Data", "resources", "categories.txt")
    CATEGORIES_TREE_FILE = os.path.join("Data", "resources", "category_tree.txt")

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
    PERSIST_DIR = os.path.join("Data", "vector_store")
    EMBEDDING_MODEL = "text-embedding-3-small"
    DISTANCE_METRIC = "cosine"

    # Chunking
    CHUNK_SIZE = 2000
    CHUNK_OVERLAP = 200

    # Embeddings
    USE_LOCAL_EMBEDDINGS = False  # Set to False for OpenAI (production/Streamlit)

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
    CHAT_HISTORY_FILE = os.path.join("Data", "chat_history.json")

    # Retrieval
    CONFIDENCE_THRESHOLD = 0.4  # Minimum similarity score for retrieved chunks to be included as context

    # LLM (OpenAI)
    LLM_MODEL_ID = "gpt-4o-mini"
    LLM_MAX_TOKENS = 1024
    SUMMARY_MAX_TOKENS = 200
