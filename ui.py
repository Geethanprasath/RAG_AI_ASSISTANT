import streamlit as st
import os

from document_loader import load_file
from chunker import clean_text, chunk_text
from retriever import (
    clear_collection,
    store_chunks,
    retrieve_chunks
)
from chatbot import generate_answer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "total_documents" not in st.session_state:
    st.session_state.total_documents = 0

if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0

if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

if "average_chunk_size" not in st.session_state:
    st.session_state.average_chunk_size = 0.0

if "indexed" not in st.session_state:
    st.session_state.indexed = False


# ============================================================
# SUPPORTED FILES
# ============================================================

def get_supported_files():

    data_folder = "data"

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    allowed_extensions = (
        ".pdf",
        ".docx",
        ".txt"
    )

    files = []

    for filename in os.listdir(data_folder):

        file_path = os.path.join(
            data_folder,
            filename
        )

        if os.path.isfile(file_path):

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension in allowed_extensions:
                files.append(file_path)

    return sorted(files)


# ============================================================
# CONFIDENCE
# ============================================================

def get_confidence(distance, answer):

    answer_lower = answer.lower()

    # --------------------------------------------------------
    # If answer says information was not found,
    # always return LOW confidence.
    # --------------------------------------------------------

    not_found_phrases = [
        "could not find",
        "couldn't find",
        "cannot find",
        "can't find",
        "not found",
        "not present in the document",
        "not present in the documents",
        "information is not available",
        "information is unavailable",
        "not available in the document",
        "not available in the documents",
        "does not contain",
        "do not contain"
    ]

    for phrase in not_found_phrases:

        if phrase in answer_lower:
            return "🔴 Low Confidence"

    # --------------------------------------------------------
    # Retrieval-based confidence
    # --------------------------------------------------------

    if distance < 0.95:

        return "🟢 High Confidence"

    elif distance < 1.20:

        return "🟡 Medium Confidence"

    else:

        return "🔴 Low Confidence"


# ============================================================
# REINDEX DOCUMENTS
# ============================================================

def reindex_documents():

    files = get_supported_files()

    if not files:

        return (
            False,
            0,
            0,
            0,
            0
        )

    all_chunks = []
    all_sources = []
    all_pages = []
    all_chunk_numbers = []

    total_pages = 0

    # --------------------------------------------------------
    # Process every file
    # --------------------------------------------------------

    for file_path in files:

        filename = os.path.basename(
            file_path
        )

        try:

            pages = load_file(
                file_path
            )

        except Exception as e:

            st.warning(
                f"⚠️ Could not read "
                f"{filename}: {e}"
            )

            continue

        total_pages += len(pages)

        # ----------------------------------------------------
        # Process each page separately
        # ----------------------------------------------------

        for page_number, page_text in enumerate(
            pages,
            start=1
        ):

            if not page_text:
                continue

            cleaned_text = clean_text(
                page_text
            )

            if not cleaned_text.strip():
                continue

            chunks = chunk_text(
                cleaned_text
            )

            # ------------------------------------------------
            # Store page + chunk metadata
            # ------------------------------------------------

            for chunk_number, chunk in enumerate(
                chunks,
                start=1
            ):

                if not chunk.strip():
                    continue

                all_chunks.append(
                    chunk
                )

                all_sources.append(
                    filename
                )

                all_pages.append(
                    page_number
                )

                all_chunk_numbers.append(
                    chunk_number
                )

    # --------------------------------------------------------
    # No chunks
    # --------------------------------------------------------

    if not all_chunks:

        return (
            False,
            len(files),
            total_pages,
            0,
            0
        )

    # --------------------------------------------------------
    # Clear old ChromaDB
    # --------------------------------------------------------

    clear_collection()

    # --------------------------------------------------------
    # Store new chunks
    # --------------------------------------------------------

    store_chunks(
        all_chunks,
        all_sources,
        all_pages,
        all_chunk_numbers
    )

    # --------------------------------------------------------
    # Average chunk size
    # --------------------------------------------------------

    average_chunk_size = (
        sum(
            len(chunk)
            for chunk in all_chunks
        )
        / len(all_chunks)
    )

    return (
        True,
        len(files),
        total_pages,
        len(all_chunks),
        average_chunk_size
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🤖 RAG Assistant")

    st.divider()

    # ========================================================
    # DOCUMENTS
    # ========================================================

    st.subheader("📚 Documents")

    files = get_supported_files()

    if files:

        for file_path in files:

            filename = os.path.basename(
                file_path
            )

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension == ".pdf":

                icon = "📕"

            elif extension == ".docx":

                icon = "📘"

            elif extension == ".txt":

                icon = "📄"

            else:

                icon = "📄"

            st.write(
                f"{icon} {filename}"
            )

    else:

        st.info(
            "No PDF, DOCX or TXT files "
            "found in data folder."
        )

    st.divider()

    # ========================================================
    # STATISTICS
    # ========================================================

    st.subheader("📊 Statistics")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Documents",
            st.session_state.total_documents
        )

        st.metric(
            "Chunks",
            st.session_state.total_chunks
        )

    with col2:

        st.metric(
            "Pages",
            st.session_state.total_pages
        )

        st.metric(
            "Avg Chunk",
            f"{st.session_state.average_chunk_size:.2f}"
        )

    st.divider()

    # ========================================================
    # ACTIONS
    # ========================================================

    st.subheader("⚙️ Actions")

    # --------------------------------------------------------
    # Clear conversation
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    # --------------------------------------------------------
    # Reindex documents
    # --------------------------------------------------------

    if st.button(
        "🔄 Reindex Documents",
        use_container_width=True
    ):

        with st.spinner(
            "Reindexing documents..."
        ):

            try:

                (
                    success,
                    total_documents,
                    total_pages,
                    total_chunks,
                    average_chunk_size
                ) = reindex_documents()

                if success:

                    st.session_state.total_documents = (
                        total_documents
                    )

                    st.session_state.total_pages = (
                        total_pages
                    )

                    st.session_state.total_chunks = (
                        total_chunks
                    )

                    st.session_state.average_chunk_size = (
                        average_chunk_size
                    )

                    st.session_state.indexed = True

                    st.success(
                        "Documents reindexed successfully!"
                    )

                    st.rerun()

                else:

                    st.error(
                        "No usable document content found."
                    )

            except Exception as e:

                st.error(
                    f"❌ Reindexing failed: {e}"
                )


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🤖 RAG Assistant")

st.caption(
    "Ask questions about your PDF, DOCX and TXT documents."
)


# ============================================================
# WELCOME MESSAGE
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
        ### 👋 Welcome!

        Upload or place your documents inside the `data` folder,
        then click **🔄 Reindex Documents**.

        You can ask questions across multiple documents.
        """
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        # ----------------------------------------------------
        # Display assistant sources
        # ----------------------------------------------------

        if (
            message["role"] == "assistant"
            and "sources" in message
        ):

            sources = message["sources"]

            if sources:

                st.markdown(
                    "#### 📚 Sources"
                )

                for i, source in enumerate(
                    sources,
                    start=1
                ):

                    with st.expander(
                        f"Source {i} — "
                        f"{source['file']}"
                    ):

                        st.write(
                            f"**File:** "
                            f"`{source['file']}`"
                        )

                        st.write(
                            f"**Page:** "
                            f"{source['page']}"
                        )

                        st.write(
                            f"**Chunk:** "
                            f"{source['chunk']}"
                        )

                        st.write(
                            f"**Distance:** "
                            f"{source['distance']:.4f}"
                        )

                        st.caption(
                            source["document"]
                        )

                # ------------------------------------------------
                # Confidence
                # ------------------------------------------------

                st.write(
                    f"**Confidence:** "
                    f"{message.get('confidence', 'Unknown')}"
                )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask anything about your documents..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # Add user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )

    # --------------------------------------------------------
    # Retrieve relevant chunks
    # --------------------------------------------------------

    try:

        (
            documents,
            distances,
            metadatas
        ) = retrieve_chunks(
            question,
            top_k=5
        )

    except Exception as e:

        st.error(
            f"❌ Retrieval error: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # No results
    # --------------------------------------------------------

    if not documents:

        answer = (
            "I couldn't find relevant information "
            "in the uploaded documents."
        )

        confidence = (
            "🔴 Low Confidence"
        )

        source_data = []

    else:

        # ----------------------------------------------------
        # Build document context
        # ----------------------------------------------------

        context = ""

        for i, (
            document,
            metadata
        ) in enumerate(
            zip(
                documents,
                metadatas
            ),
            start=1
        ):

            source = metadata.get(
                "source",
                "Unknown"
            )

            page = metadata.get(
                "page",
                "Unknown"
            )

            chunk = metadata.get(
                "chunk",
                "Unknown"
            )

            context += (

                f"\n\n"
                f"Source {i}\n"
                f"File: {source}\n"
                f"Page: {page}\n"
                f"Chunk: {chunk}\n"
                f"Content:\n"
                f"{document}"
            )

        # ----------------------------------------------------
        # Conversation history
        # ----------------------------------------------------

        conversation_history = ""

        for message in st.session_state.messages[:-1]:

            conversation_history += (

                f"{message['role'].upper()}: "
                f"{message['content']}\n"
            )

        # ----------------------------------------------------
        # Add history to context
        # ----------------------------------------------------

        if conversation_history:

            context = (

                "Previous conversation:\n"

                + conversation_history

                + "\n\n"

                + "Relevant document information:\n"

                + context
            )

        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        try:

            answer = generate_answer(
                question,
                context
            )

        except Exception as e:

            st.error(
                f"❌ LLM error: {e}"
            )

            st.stop()

        # ----------------------------------------------------
        # IMPORTANT:
        # Confidence is calculated AFTER answer
        # ----------------------------------------------------

        best_distance = distances[0]

        confidence = get_confidence(
            best_distance,
            answer
        )

        # ----------------------------------------------------
        # Source data
        # ----------------------------------------------------

        source_data = []

        for (
            document,
            distance,
            metadata
        ) in zip(
            documents,
            distances,
            metadatas
        ):

            source_data.append(
                {
                    "file": metadata.get(
                        "source",
                        "Unknown"
                    ),

                    "page": metadata.get(
                        "page",
                        "Unknown"
                    ),

                    "chunk": metadata.get(
                        "chunk",
                        "Unknown"
                    ),

                    "distance": distance,

                    "document": document
                }
            )


    # ========================================================
    # DISPLAY ASSISTANT
    # ========================================================

    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            answer
        )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        st.markdown(
            f"**Confidence:** {confidence}"
        )

        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        if source_data:

            st.markdown(
                "#### 📚 Sources"
            )

            for i, source in enumerate(
                source_data,
                start=1
            ):

                with st.expander(
                    f"Source {i} — "
                    f"{source['file']}"
                ):

                    st.write(
                        f"**File:** "
                        f"`{source['file']}`"
                    )

                    st.write(
                        f"**Page:** "
                        f"{source['page']}"
                    )

                    st.write(
                        f"**Chunk:** "
                        f"{source['chunk']}"
                    )

                    st.write(
                        f"**Distance:** "
                        f"{source['distance']:.4f}"
                    )

                    st.caption(
                        source["document"]
                    )


    # ========================================================
    # SAVE ASSISTANT MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "confidence": confidence,
            "sources": source_data
        }
    )