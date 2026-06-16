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
    pip install langchain-openai chromadb
"""

import json
import os
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from Source.Constants import Constants
from Source.Prompts import Prompts


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
        """Initialise the OpenAI LLM client via LangChain."""
        self.llm = ChatOpenAI(
            model=Constants.LLM_MODEL_ID,
            max_tokens=Constants.LLM_MAX_TOKENS,
        )

    def _print(self, msg):
        """Print a message only when VERBOSE mode is enabled."""
        if Constants.VERBOSE:
            print(msg)

    # ==========================================================================
    # STEP 1 — SELF-QUERY (LLM-POWERED CATEGORY DETECTION)
    # ==========================================================================

    def _load_category_tree(self):
        """Load the category tree string from Resources."""
        if not os.path.exists(Constants.CATEGORIES_TREE_FILE):
            return ""
        with open(Constants.CATEGORIES_TREE_FILE, "r", encoding="utf-8") as f:
            return f.read()

    def _load_category_list(self):
        """Load valid category paths from Resources."""
        if not os.path.exists(Constants.CATEGORIES_LIST_FILE):
            return []
        with open(Constants.CATEGORIES_LIST_FILE, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def detect_categories(self, query):
        """
        Use the LLM to analyse the query against the category tree and extract:
        - A refined semantic search query
        - Relevant category path filters

        Includes recent conversation context so follow-up questions
        can be correctly categorised.

        Returns:
            tuple[str, list[str]]: (refined_query, list of category paths)
        """
        category_tree = self._load_category_tree()
        if not category_tree:
            return query, []

        valid_paths = self._load_category_list()

        # Include recent conversation so the self-query LLM can resolve
        # ambiguous follow-up questions (e.g. "how much do I need?" after
        # discussing SA investment → correctly maps to investment category)
        history_context = ""
        if self.history["recent"]:
            recent_turns = self.history["recent"][-4:]  # Last 2 exchanges
            for turn in recent_turns:
                role = "User" if turn["role"] == "user" else "Assistant"
                history_context += f"{role}: {turn['content'][:200]}\n"

        user_message = ""
        if history_context:
            user_message += f"Conversation context:\n{history_context}\n"
        user_message += f"Current question: {query}"

        try:
            prompt = Prompts.SELF_QUERY_PROMPT.format(category_tree=category_tree)
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=user_message),
            ]
            response = self.llm.invoke(messages, max_tokens=200)
            result = json.loads(response.content)

            refined_query = result.get("query", query)
            categories = result.get("categories", [])

            # Validate: keep only categories that match valid paths (prefix match)
            validated = []
            for cat in categories:
                for valid in valid_paths:
                    if valid.startswith(cat):
                        validated.append(cat)
                        break

            self._print(f"  Self-query: '{refined_query}'")
            self._print(f"  Categories: {validated if validated else 'All (no filter)'}")
            return refined_query, validated

        except Exception as e:
            self._print(f"  Self-query failed: {e}, falling back to raw query")
            return query, []

    # ==========================================================================
    # STEP 2 — VECTOR DB SEARCH
    # ==========================================================================

    def search(self, query, categories=None):
        """
        Retrieve the most relevant chunks from the vector database.

        Expands category prefixes into all matching paths for $in filtering.

        Args:
            query: The semantic search query (possibly refined by self-query).
            categories: Optional list of category path prefixes to filter on.

        Returns:
            list[dict]: Relevant chunks with keys: text, url, category,
                        chunk_index, score.
        """
        # Build category filter by expanding prefixes to all matching paths
        where_filter = None
        if categories:
            valid_paths = self._load_category_list()
            matching = [p for p in valid_paths if any(p.startswith(cat) for cat in categories)]
            if matching:
                where_filter = {"category": {"$in": matching}}

        results = self.vdb.query(
            query_text=query,
            top_k=Constants.DEFAULT_TOP_K,
            where_filter=where_filter,
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
                f"[Source {i}: {chunk['category']} | {chunk['url']}]\n{chunk['text']}"
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
            "system": Prompts.SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_message}],
        }

    # ==========================================================================
    # STEP 5 — LLM GENERATION
    # ==========================================================================

    def generate(self, prompt):
        """
        Call the LLM (OpenAI via LangChain) to generate a response.

        Args:
            prompt: Dict with 'system' and 'messages' keys from build_prompt().

        Returns:
            str: The LLM's generated response text.
        """
        try:
            messages = [
                SystemMessage(content=prompt["system"]),
                HumanMessage(content=prompt["messages"][0]["content"]),
            ]
            response = self.llm.invoke(messages)
            return response.content

        except Exception as e:
            self._print(f"  LLM Error: {e}")
            return Prompts.ERROR_MESSAGE

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
            messages = [
                SystemMessage(content=Prompts.SUMMARY_PROMPT),
                HumanMessage(content=older_text),
            ]
            response = self.llm.invoke(messages, max_tokens=Constants.SUMMARY_MAX_TOKENS)
            self.history["summary"] = response.content
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

        # Step 1 — Self-query: detect categories and refine query
        refined_query, categories = self.detect_categories(query)

        # Step 2 — Search vector DB with refined query and category filters
        chunks = self.search(refined_query, categories)

        # Step 3 — Build context
        context = self.build_context(chunks)

        # Step 4 — Build prompt
        prompt = self.build_prompt(query, context)

        # Step 5 — Generate response
        if not chunks:
            answer = Prompts.NO_ANSWER_MESSAGE
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
