"""
Chat Engine
============
Orchestrates the RAG (Retrieval-Augmented Generation) chat pipeline
for answering user questions about CPF policies.

Pipeline:
    User Query → Category Detection → Vector DB Search →
    Context Builder → Prompt Builder → LLM Generation → History Update

Usage:
    from Source.Vector_Database import VectorDatabase
    from Source.Chat_Engine import ChatEngine

    vdb = VectorDatabase()
    chat = ChatEngine(vdb)
    response = chat.ask("How do I use CPF for housing?")
    chat.reset()

Dependencies:
    pip install boto3 sentence-transformers chromadb
"""

import json
import os
import time

from Source.Constants import Constants


class ChatEngine:
    """
    RAG-based chat engine that answers questions using scraped CPF policy data.

    Each step in the pipeline is implemented as a separate method:
        1. detect_categories()  — Classify query into relevant policy categories
        2. search()             — Retrieve relevant chunks from the vector DB
        3. build_context()      — Format retrieved chunks into LLM-readable context
        4. build_prompt()       — Assemble full prompt with history and context
        5. generate()           — Call LLM to produce a response
        6. update_history()     — Store the exchange and summarise if needed
        7. reset()              — Clear conversation history
    """

    def __init__(self, vector_db):
        """
        Initialise the chat engine.

        Args:
            vector_db: An initialised VectorDatabase instance for retrieval.
        """
        self.vdb = vector_db
        self.history = {
            "summary": "",
            "recent": [],
        }
        self._init_llm_client()

    def _init_llm_client(self):
        """Initialise the Amazon Bedrock client for LLM calls."""
        import boto3
        self.bedrock = boto3.client(
            "bedrock-runtime",
            region_name=Constants.LLM_REGION,
        )

    def _print(self, msg):
        """Print a message only when VERBOSE mode is enabled."""
        if Constants.VERBOSE:
            print(msg)

    # ==========================================================================
    # STEP 1 — CATEGORY DETECTION
    # ==========================================================================

    def detect_categories(self, query):
        """
        Analyse the user's query to determine which policy categories are relevant.

        Uses keyword matching against a predefined map. Returns an empty list
        if no clear category is detected (search will span all categories).

        Args:
            query: The user's natural language question.

        Returns:
            list[str]: Matching category names, e.g. ["Housing", "Retirement"].
        """
        query_lower = query.lower()
        detected = []

        for category, keywords in Constants.CATEGORY_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                detected.append(category)

        self._print(f"  Categories detected: {detected if detected else 'All (no filter)'}")
        return detected

    # ==========================================================================
    # STEP 2 — VECTOR DB SEARCH
    # ==========================================================================

    def search(self, query, categories=None):
        """
        Retrieve the most relevant chunks from the vector database.

        Applies category filtering if categories were detected. Filters out
        results below the confidence threshold.

        Args:
            query: The user's natural language question.
            categories: Optional list of categories to filter on.

        Returns:
            list[dict]: Relevant chunks with keys: text, url, category,
                        chunk_index, score.
        """
        results = self.vdb.query(
            query_text=query,
            top_k=Constants.DEFAULT_TOP_K,
            categories=categories if categories else None,
        )

        # Filter out low-confidence results
        filtered = [r for r in results if r["score"] >= Constants.CONFIDENCE_THRESHOLD]

        self._print(f"  Chunks retrieved: {len(results)}, above threshold: {len(filtered)}")
        return filtered

    # ==========================================================================
    # STEP 3 — CONTEXT BUILDER
    # ==========================================================================

    def build_context(self, chunks):
        """
        Format retrieved chunks into a structured context string for the LLM.

        Each chunk is prefixed with its source URL for citation purposes.

        Args:
            chunks: List of chunk dicts from vector DB search.

        Returns:
            str: Formatted context string, or empty string if no chunks.
        """
        if not chunks:
            return ""

        context_parts = []
        for i, chunk in enumerate(chunks, start=1):
            context_parts.append(
                f"[Source {i}: {chunk['url']}]\n{chunk['text']}"
            )

        return "\n\n".join(context_parts)

    # ==========================================================================
    # STEP 4 — PROMPT BUILDER
    # ==========================================================================

    def build_prompt(self, query, context):
        """
        Assemble the full LLM prompt including system instructions,
        conversation history (summary + recent turns), context, and query.

        Args:
            query: The user's current question.
            context: Formatted context string from build_context().

        Returns:
            dict: Contains 'system' prompt and 'messages' list for the LLM.
        """
        # Build conversation history section
        history_text = ""
        if self.history["summary"]:
            history_text += f"Conversation summary: {self.history['summary']}\n\n"

        for turn in self.history["recent"]:
            role = "User" if turn["role"] == "user" else "Assistant"
            history_text += f"{role}: {turn['content']}\n"

        # Assemble the user message
        user_message = ""
        if history_text:
            user_message += f"Previous conversation:\n{history_text}\n"
        if context:
            user_message += f"Context:\n{context}\n\n"
        else:
            user_message += "Context: No relevant information found.\n\n"
        user_message += f"Question: {query}"

        return {
            "system": Constants.SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_message}],
        }

    # ==========================================================================
    # STEP 5 — LLM GENERATION
    # ==========================================================================

    def generate(self, prompt):
        """
        Call the LLM (Amazon Bedrock) to generate a response.

        Args:
            prompt: Dict with 'system' and 'messages' keys from build_prompt().

        Returns:
            str: The LLM's generated response text.
        """
        try:
            response = self.bedrock.invoke_model(
                modelId=Constants.LLM_MODEL_ID,
                body=json.dumps({
                    "system": prompt["system"],
                    "messages": prompt["messages"],
                    "max_tokens": Constants.LLM_MAX_TOKENS,
                    "anthropic_version": "bedrock-2023-05-31",
                }),
            )

            result = json.loads(response["body"].read())
            return result["content"][0]["text"]

        except Exception as e:
            self._print(f"  LLM Error: {e}")
            return Constants.ERROR_MESSAGE

    # ==========================================================================
    # STEP 6 — HISTORY UPDATE
    # ==========================================================================

    def update_history(self, query, response, sources=None):
        """
        Append the latest exchange to conversation history.

        If history exceeds MAX_HISTORY_TURNS, older turns are summarised
        into a compressed summary string to preserve context without
        overloading the token budget.

        Args:
            query: The user's question.
            response: The assistant's generated response.
            sources: Optional list of source URLs cited in the response.
        """
        # Append new exchange
        self.history["recent"].append({"role": "user", "content": query})
        self.history["recent"].append({
            "role": "assistant",
            "content": response,
            "sources": sources or [],
        })

        # Summarise if history exceeds the limit
        if len(self.history["recent"]) > Constants.MAX_HISTORY_TURNS * 2:
            self._summarise_history()

    def _summarise_history(self):
        """
        Compress older conversation turns into a summary.

        Keeps the most recent MAX_HISTORY_TURNS exchanges verbatim and
        summarises everything before that into a condensed string using the LLM.
        """
        # Split: older turns to summarise, recent turns to keep
        keep_count = Constants.MAX_HISTORY_TURNS * 2  # 2 messages per turn (user + assistant)
        older_turns = self.history["recent"][:-keep_count]
        recent_turns = self.history["recent"][-keep_count:]

        # Format older turns for summarisation
        older_text = ""
        for turn in older_turns:
            role = "User" if turn["role"] == "user" else "Assistant"
            older_text += f"{role}: {turn['content']}\n"

        # Include existing summary for continuity
        if self.history["summary"]:
            older_text = f"Previous summary: {self.history['summary']}\n\n{older_text}"

        # Ask LLM to summarise
        try:
            summary_response = self.bedrock.invoke_model(
                modelId=Constants.LLM_MODEL_ID,
                body=json.dumps({
                    "system": Constants.SUMMARY_PROMPT,
                    "messages": [{"role": "user", "content": older_text}],
                    "max_tokens": Constants.SUMMARY_MAX_TOKENS,
                    "anthropic_version": "bedrock-2023-05-31",
                }),
            )
            result = json.loads(summary_response["body"].read())
            self.history["summary"] = result["content"][0]["text"]
        except Exception as e:
            # Fallback: simple truncation if LLM call fails
            self._print(f"  Summary Error: {e}")
            self.history["summary"] += f" User discussed: {older_text[:200]}..."

        # Keep only recent turns
        self.history["recent"] = recent_turns
        self._print(f"  History summarised. Recent turns: {len(recent_turns)}")

    # ==========================================================================
    # STEP 7 — RESET
    # ==========================================================================

    def reset(self):
        """
        Clear all conversation history and start a fresh session.

        Removes both the summary and recent messages. Optionally archives
        the old history to a JSON file before clearing.
        """
        # Optionally archive before clearing
        if self.history["recent"]:
            self._archive_history()

        self.history = {
            "summary": "",
            "recent": [],
        }
        self._print("  Chat history reset.")

    def _archive_history(self):
        """
        Save the current conversation to a JSON file for record keeping.

        Appends to existing archive file with a timestamp.
        """
        archive_path = Constants.CHAT_HISTORY_FILE
        archive_entry = {
            "archived_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": self.history["summary"],
            "messages": self.history["recent"],
        }

        # Load existing archive or start fresh
        archive = []
        if os.path.exists(archive_path):
            try:
                with open(archive_path, "r", encoding="utf-8") as f:
                    archive = json.load(f)
            except (json.JSONDecodeError, Exception):
                archive = []

        archive.append(archive_entry)

        with open(archive_path, "w", encoding="utf-8") as f:
            json.dump(archive, f, indent=2, ensure_ascii=False)

        self._print(f"  History archived to {archive_path}")

    # ==========================================================================
    # ORCHESTRATOR
    # ==========================================================================

    def ask(self, query):
        """
        Execute the full chat pipeline for a user query.

        Pipeline:
            1. Detect categories from the query
            2. Search vector DB for relevant chunks
            3. Build context from retrieved chunks
            4. Build the full LLM prompt with history
            5. Generate a response via LLM
            6. Update conversation history

        Args:
            query: The user's natural language question.

        Returns:
            dict: Contains 'answer' (str) and 'sources' (list of URLs).
        """
        self._print(f"\n  Query: {query}")

        # Step 1 — Detect categories
        categories = self.detect_categories(query)

        # Step 2 — Search vector DB
        chunks = self.search(query, categories)

        # Step 3 — Build context
        context = self.build_context(chunks)

        # Step 4 — Build prompt
        prompt = self.build_prompt(query, context)

        # Step 5 — Generate response
        if not chunks:
            answer = Constants.NO_ANSWER_MESSAGE
        else:
            answer = self.generate(prompt)

        # Extract source URLs from retrieved chunks
        sources = list({chunk["url"] for chunk in chunks})

        # Step 6 — Update history
        self.update_history(query, answer, sources)

        return {
            "answer": answer,
            "sources": sources,
        }
