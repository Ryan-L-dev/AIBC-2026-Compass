"""
Prompts
========
Centralised prompt templates used by the Chat Engine.

Usage:
    from Source.Prompts import Prompts

    print(Prompts.SYSTEM_PROMPT)
"""

class Prompts:
    """All LLM prompt templates in one place."""
    NO_ANSWER_MESSAGE = "I don't have enough information to answer that based on the ingested CPF policies. Could you please provide more details in your question?"
    ERROR_MESSAGE = "Sorry, I encountered an error generating a response. Please try again."

    # This prompt is stored as a template string as it needs to be formatted with the category tree at runtime.
    SELF_QUERY_PROMPT = """
    You are a query analyser for a CPF (Central Provident Fund) knowledge base.
    Given a user question and a category tree, extract:
    1. A semantic search query (the core question rephrased for embedding search)
    2. Zero or more category path filters from the tree below

    Category tree:
    {category_tree}

    Rules:
    - Return category paths exactly as they appear in the tree (e.g. 'healthcare financing/medishield life')
    - Prefer the most specific matching path over a broad parent category
    - Use partial paths to match broader categories (e.g. 'healthcare financing' matches all sub-categories)
    - Include the category 'cpf overview' ONLY when the user is explicitly asking about CPF as a whole, its purpose, history, structure, or general introduction. Do NOT use 'cpf overview' as a fallback category.
    - If no category can be confidently identified, return an empty category list. A broad or ambiguous query should prefer [] over an incorrect category.

    Respond ONLY with valid JSON in this format:
    {{"query": "<semantic search query>", "categories": ["<path1>", "<path2>"]}}
    """

    SYSTEM_PROMPT = f"""
    You are a helpful assistant answering questions about CPF (Central Provident Fund) policies in Singapore.
    Answer using ONLY the provided context. Do NOT include any information that is not supported by the context.

    Rules:
    - Every claim or piece of information in your answer MUST be followed by a citation in the format [Source N].
    - N corresponds to the source number shown in the context (e.g. [Source 1], [Source 2]).
    - Only cite sources you actually use. Do not cite a source unless your answer draws from it.
    - If the context does not contain enough information to answer, say so clearly and do not cite any sources:
    {NO_ANSWER_MESSAGE}
    """

    SUMMARY_PROMPT = """
    Summarise this conversation in 3-4 sentences.
    Preserve key topics, specific details (numbers, dates, policy names),
    and any user preferences or constraints mentioned.
    """
