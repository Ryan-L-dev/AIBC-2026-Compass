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

from Source.Data_Scraper import DataScraper
from Source.Vector_Database import VectorDatabase
from Source.Chat_Engine import ChatEngine

def data_preprocessing():
    # Step 1 — Scrape website content
    scraper = DataScraper()
    scraper.process()

    # Step 2 — Embed into vector database
    vdb = VectorDatabase()
    vdb.process()

def chat_interface(vdb):
    # Step 3 — Chat (placeholder interactive loop)
    chat = ChatEngine(vdb)

    print("\nCPF Policy Assistant (type 'quit' to exit, 'reset' to clear history)")
    print("-" * 60)

    while True:
        query = input("\nYou: ").strip()

        if not query:
            continue
        if query.lower() == "quit":
            break
        if query.lower() == "reset":
            chat.reset()
            print("  Chat history cleared.")
            continue

        result = chat.ask(query)
        print(f"\nAssistant: {result['answer']}")

        if result["sources"]:
            print("\n  Sources:")
            for url in result["sources"]:
                print(f"    - {url}")

def main():
    #data_preprocessing()
    #chat_interface()
    vdb = VectorDatabase()
    output = vdb.query("How do I use CPF for housing?")
    [print(i) for i in output]
    pass

if __name__ == "__main__":
    main()
