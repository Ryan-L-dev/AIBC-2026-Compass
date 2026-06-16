"""
Main Entry Point
=================
Orchestrates the full pipeline:
    1. Scrape CPF website content
    2. Embed scraped content into the vector database
    3. Start chat engine for policy Q&A

Usage:
    python main.py
"""

import os

from Source.Data_Scraper import DataScraper
from Source.Vector_Database import VectorDatabase
#from Source.Chat_Engine import ChatEngine

def data_preprocessing():
    # Step 1 — Scrape website content
    scraper = DataScraper()
    scraper.process()

    # Step 2 — Embed into vector database
    vdb = VectorDatabase()
    vdb.process()

# def chat_interface(vdb):
#     # Step 3 — Chat (placeholder interactive loop)
#     chat = ChatEngine(vdb)

#     print("\nCPF Policy Assistant (type 'quit' to exit, 'reset' to clear history)")
#     print("-" * 60)

#     while True:
#         query = input("\nYou: ").strip()

#         if not query:
#             continue
#         if query.lower() == "quit":
#             break
#         if query.lower() == "reset":
#             chat.reset()
#             print("  Chat history cleared.")
#             continue

#         result = chat.ask(query)
#         print(f"\nAssistant: {result['answer']}")

#         if result["sources"]:
#             print("\n  Sources:")
#             for url in result["sources"]:
#                 print(f"    - {url}")

def _expand_categories(prefixes):
    """Expand category prefixes into all matching paths from the categories list."""
    from Source.Constants import Constants
    if not os.path.exists(Constants.CATEGORIES_LIST_FILE):
        return prefixes
    with open(Constants.CATEGORIES_LIST_FILE, "r", encoding="utf-8") as f:
        valid_paths = [line.strip() for line in f if line.strip()]
    return [p for p in valid_paths if any(p.startswith(cat) for cat in prefixes)]


def test_vector_db():
    """Test vector database queries with and without category filters."""
    vdb = VectorDatabase()

    # Test: semantic search (no filter)
    print("\n--- Query: How do I use CPF for housing? ---")
    output = vdb.query("How do I use CPF for housing?")
    [print(i) for i in output]

    # Test: semantic search with category filter
    categories = ["healthcare financing"]
    expanded = _expand_categories(categories)
    print(f"\n--- Query: What insurance covers me? ---")
    print(f"  Categories: {categories}")
    print(f"  Expanded:   {expanded}")
    output = vdb.query(
        "What insurance covers me?",
        where_filter={"category": {"$in": expanded}},
    )
    [print(i) for i in output]

    # Test: semantic search with root category (expands to all sub-categories)
    categories = ["growing your savings"]
    expanded = _expand_categories(categories)
    print(f"\n--- Query: Can I invest my special account? ---")
    print(f"  Categories: {categories}")
    print(f"  Expanded:   {expanded}")
    output = vdb.query(
        "Can I invest my special account?",
        where_filter={"category": {"$in": expanded}},
    )
    [print(i) for i in output]


def main():
    #data_preprocessing()
    #chat_interface()
    test_vector_db()

if __name__ == "__main__":
    main()
