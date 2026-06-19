"""
Streamlit Frontend
===================
Two-tab interface for the CPF Policy Assistant:
    - Chat: Ask questions about CPF policies
    - Data Management: Re-scrape and re-embed on demand

Usage:
    streamlit run app.py
"""

import os
import streamlit as st
from Source.Constants import Constants

st.set_page_config(page_title="CPF Policy Assistant", page_icon="💬", layout="wide")

# ==========================================================================
# AUTHENTICATION
# ==========================================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("💬 CPF Policy Assistant")
    st.markdown("Please log in to continue.")

    with st.form("login_form"):
        user_id = st.text_input("User ID")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

        if submitted:
            if user_id == "Admin" and password == "Beacon":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials.")
    st.stop()

# ==========================================================================
# MAIN APP (shown only after login)
# ==========================================================================

st.title("💬 CPF Policy Assistant")

# ==========================================================================
# SESSION STATE INITIALISATION
# ==========================================================================

if "vdb" not in st.session_state:
    from Source.Vector_Database import VectorDatabase
    st.session_state.vdb = VectorDatabase()

if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================================================
# TABS
# ==========================================================================

tab_chat, tab_data = st.tabs(["💬 CPF Chat", "🔄 Data Management"])

# ==========================================================================
# TAB 1 — CHAT
# ==========================================================================

with tab_chat:
    st.markdown("Ask questions about CPF policies and receive grounded answers with source citations.")
    if st.button("🗑️ Clear Chat"):
        if st.session_state.chat_engine:
            st.session_state.chat_engine.reset()
        st.session_state.messages = []
        st.rerun()

    # Display chat history
    chat_history_container = st.container()
    with chat_history_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("Sources"):
                        for url in msg["sources"]:
                            st.markdown(f"- [{url}]({url})")
    
    # Chat input - always at the bottom
    query = st.chat_input("Ask about CPF policies...")
    
    # Generate responses immediately when query is entered
    if query:
        # Initialise chat engine on first use
        if st.session_state.chat_engine is None:
            from Source.Chat_Engine import ChatEngine
            st.session_state.chat_engine = ChatEngine(st.session_state.vdb)

        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Generate response
        result = st.session_state.chat_engine.ask(query)
        
        # Store response
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })
        st.rerun()

# ==========================================================================
# TAB 2 — DATA MANAGEMENT
# ==========================================================================

with tab_data:
    st.subheader("Data Pipeline")
    st.markdown("Re-scrape the CPF website and update the vector database.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Scrape & Embed")
        st.caption("Scrapes cpf.gov.sg and embeds changed pages into the vector database.")
        if st.button("▶️ Scrape & Embed", use_container_width=True):
            from Source.Data_Scraper import DataScraper
            import pandas as pd

            # Step 1 — Scrape
            scraper = DataScraper()
            progress = st.progress(0, text="Scraping...")
            scraper.process(progress_callback=lambda cur, tot: progress.progress(cur / tot, text=f"Scraping {cur}/{tot}"))

            # Scrape summary
            log_df = pd.DataFrame(scraper.log)
            attempted = len(log_df)
            scraped = len(log_df[log_df[Constants.COL_SCRAPED] == Constants.SCRAPED_YES])
            skipped = attempted - scraped

            # Step 2 — Embed
            vdb = st.session_state.vdb
            progress.progress(0, text="Embedding...")
            vdb.process(progress_callback=lambda cur, tot: progress.progress(cur / tot, text=f"Embedding {cur}/{tot}"))

            progress.empty()
            count = vdb.vectorstore._collection.count()
            st.success(f"Pipeline complete. Scraped: {scraped}/{attempted} pages ({skipped} skipped). Vectors in DB: {count}")

    with col2:
        st.markdown("#### Embed Only")
        st.caption("Re-embed existing scraped files without re-scraping.")
        if st.button("▶️ Run Embedding", use_container_width=True):
            vdb = st.session_state.vdb
            progress = st.progress(0, text="Embedding...")
            stats = vdb.process(progress_callback=lambda cur, tot: progress.progress(cur / tot, text=f"Embedding {cur}/{tot}"))
            progress.empty()
            count = vdb.vectorstore._collection.count()
            if stats:
                st.success(f"Embedding complete. Embedded: {stats['embedded']}/{stats['files_found']} files ({stats['skipped']} unchanged). Vectors in DB: {count}")
            else:
                st.success(f"Embedding complete. All files up to date. Vectors in DB: {count}")

    st.divider()

    # Status display
    st.markdown("#### Current Status")
    col_a, col_b, col_c = st.columns(3)

    scraped_count = sum(1 for _, _, files in os.walk(Constants.OUTPUT_DIR) for f in files if f.endswith(".md"))
    vector_count = st.session_state.vdb.vectorstore._collection.count()

    col_a.metric("Scraped Pages", scraped_count)
    col_b.metric("Vectors in DB", vector_count)
    col_c.metric("Embedding Model", "Local (ONNX)" if Constants.USE_LOCAL_EMBEDDINGS else "OpenAI")
