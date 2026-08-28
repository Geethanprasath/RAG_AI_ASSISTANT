import os

from document_loader import load_file
from chunker import clean_text, chunk_text, save_chunks

from retriever import (
    clear_collection,
    store_chunks,
    retrieve_chunks
)

from chatbot import generate_answer


# ============================================
# Confidence
# ============================================

def get_confidence(distance, answer):

    answer_lower = answer.lower()

    # If the LLM says the information was not found,
    # always show Low Confidence.
    not_found_phrases = [
        "could not find",
        "couldn't find",
        "cannot find",
        "can't find",
        "not found",
        "information is not available",
        "information is unavailable",
        "not present in the documents",
        "not present in the document",
        "does not contain",
        "do not contain"
    ]

    for phrase in not_found_phrases:
        if phrase in answer_lower:
            return "🔴 Low Confidence"

    # Confidence based on retrieval distance
    if distance < 1.00:
        return "🟢 High Confidence"

    elif distance < 1.50:
        return "🟡 Medium Confidence"

    else:
        return "🔴 Low Confidence"


# ============================================
# Main
# ============================================

def main():

    print("=" * 60)
    print("              🤖 RAG ASSISTANT")
    print("=" * 60)

    # ========================================
    # Data folder
    # ========================================

    data_folder = "data"

    if not os.path.exists(data_folder):

        print("\n❌ data folder not found.")
        return

    # ========================================
    # Find all supported files
    # ========================================

    allowed_extensions = [
        ".pdf",
        ".docx",
        ".txt"
    ]

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

    # ========================================
    # Check files
    # ========================================

    if not files:

        print(
            "\n❌ No PDF, DOCX or TXT files "
            "found in data folder."
        )

        return

    print("\nDocuments found:")

    for file_path in files:

        print(
            f"  ✅ {os.path.basename(file_path)}"
        )

    print(
        f"\nTotal files: {len(files)}"
    )

    # ========================================
    # Process ALL documents
    # ========================================

    all_chunks = []
    all_sources = []
    all_pages = []
    all_chunk_numbers = []

    total_pages = 0

    successful_files = 0
    skipped_files = 0

    skipped_file_names = []

    print("\n" + "=" * 60)
    print("             LOADING DOCUMENTS")
    print("=" * 60)

    for file_path in files:

        file_name = os.path.basename(
            file_path
        )

        print(
            f"\n📄 Processing: {file_name}"
        )

        # ------------------------------------
        # Load file
        # ------------------------------------

        try:

            pages = load_file(
                file_path
            )

        except Exception as e:

            print(
                f"❌ Error reading "
                f"{file_name}: {e}"
            )

            skipped_files += 1
            skipped_file_names.append(
                file_name
            )

            continue

        # ------------------------------------
        # Empty file check
        # ------------------------------------

        if not pages:

            print(
                f"⚠️ {file_name} is empty."
            )

            skipped_files += 1
            skipped_file_names.append(
                file_name
            )

            continue

        print(
            f"   Pages: {len(pages)}"
        )

        total_pages += len(pages)

        file_chunk_count = 0

        # ------------------------------------
        # Process page by page
        # ------------------------------------

        for page_number, page_text in enumerate(
            pages,
            start=1
        ):

            # Skip empty page
            if not page_text.strip():
                continue

            cleaned_text = clean_text(
                page_text
            )

            # Skip page with no usable text
            if not cleaned_text.strip():
                continue

            page_chunks = chunk_text(
                cleaned_text
            )

            # --------------------------------
            # Store chunks
            # --------------------------------

            for chunk_number, chunk in enumerate(
                page_chunks,
                start=1
            ):

                if not chunk.strip():
                    continue

                all_chunks.append(
                    chunk
                )

                all_sources.append(
                    file_name
                )

                all_pages.append(
                    page_number
                )

                all_chunk_numbers.append(
                    chunk_number
                )

                file_chunk_count += 1

        print(
            f"   Chunks: {file_chunk_count}"
        )

        # ------------------------------------
        # Check if file produced chunks
        # ------------------------------------

        if file_chunk_count == 0:

            print(
                f"⚠️ {file_name} contains "
                f"no usable text."
            )

            skipped_files += 1
            skipped_file_names.append(
                file_name
            )

        else:

            successful_files += 1

    # ========================================
    # Document statistics
    # ========================================

    print("\n" + "=" * 60)

    print(
        f"Total files found       : {len(files)}"
    )

    print(
        f"Successfully indexed    : {successful_files}"
    )

    print(
        f"Skipped files           : {skipped_files}"
    )

    print(
        f"Total pages             : {total_pages}"
    )

    print(
        f"Total chunks            : {len(all_chunks)}"
    )

    print("=" * 60)

    # ========================================
    # Show skipped files
    # ========================================

    if skipped_file_names:

        print("\n⚠️ Skipped files:")

        for name in skipped_file_names:

            print(
                f"  ❌ {name}"
            )

    # ========================================
    # Check chunks
    # ========================================

    if not all_chunks:

        print(
            "\n❌ No usable text found "
            "in the documents."
        )

        return

    # ========================================
    # Clear old ChromaDB
    # ========================================

    print(
        "\n🗑️ Clearing old ChromaDB data..."
    )

    clear_collection()

    # ========================================
    # Save chunks
    # ========================================

    save_chunks(
        all_chunks
    )

    print(
        "✅ Chunks saved."
    )

    # ========================================
    # Store embeddings
    # ========================================

    print(
        "\n🔄 Creating embeddings..."
    )

    store_chunks(
        all_chunks,
        all_sources,
        all_pages,
        all_chunk_numbers
    )

    print(
        "✅ All documents stored in ChromaDB."
    )

    # ========================================
    # Ready
    # ========================================

    print("\n" + "=" * 60)

    print(
        "          🤖 RAG ASSISTANT READY"
    )

    print("=" * 60)

    print(
        "\nSearching across:"
    )

    for file_path in files:

        print(
            f"  📄 {os.path.basename(file_path)}"
        )

    print(
        "\nType 'exit' to stop."
    )

    print(
        "Type 'clear' to clear conversation."
    )

    # ========================================
    # Conversation history
    # ========================================

    conversation_history = []

    # ========================================
    # Question loop
    # ========================================

    while True:

        question = input(
            "\nQuestion: "
        ).strip()

        # ------------------------------------
        # Exit
        # ------------------------------------

        if question.lower() == "exit":

            print(
                "\n👋 Goodbye!"
            )

            break

        # ------------------------------------
        # Clear conversation
        # ------------------------------------

        if question.lower() == "clear":

            conversation_history = []

            print(
                "\n🗑️ Conversation cleared."
            )

            continue

        # ------------------------------------
        # Empty question
        # ------------------------------------

        if not question:

            print(
                "⚠️ Please enter a question."
            )

            continue

        # ====================================
        # Retrieve from ALL documents
        # ====================================

        try:

            documents, distances, metadatas = (
                retrieve_chunks(
                    question,
                    top_k=5
                )
            )

        except Exception as e:

            print(
                f"\n❌ Retrieval error: {e}"
            )

            continue

        # ====================================
        # No retrieval result
        # ====================================

        if not documents:

            answer = (
                "I could not find this information "
                "in the documents."
            )

            print("\n" + "-" * 60)

            print("Answer:")
            print(answer)

            print(
                "\nConfidence: 🔴 Low Confidence"
            )

            print("-" * 60)

            continue

        # ====================================
        # Best retrieval distance
        # ====================================

        best_distance = distances[0]

        # ====================================
        # Build document context
        # ====================================

        document_context = ""

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

            document_context += (

                f"\nSource {i}\n"

                f"File: {source}\n"

                f"Page: {page}\n"

                f"Chunk: {chunk}\n"

                f"Text: {document}\n"
            )

        # ====================================
        # Conversation history
        # ====================================

        history_text = ""

        for item in conversation_history:

            history_text += (

                f"{item['role'].upper()}: "

                f"{item['content']}\n"
            )

        # ====================================
        # Complete context
        # ====================================

        context = (

            "Answer ONLY using the information "
            "provided in the document context.\n\n"

            "If the answer is not present in "
            "the documents, say exactly: "
            "'I could not find this information "
            "in the documents.'\n\n"
        )

        # ------------------------------------
        # Add conversation history
        # ------------------------------------

        if history_text:

            context += (

                "Previous conversation:\n"

                + history_text

                + "\n"
            )

        # ------------------------------------
        # Add document context
        # ------------------------------------

        context += (

            "Relevant document information:\n"

            + document_context
        )

        # ====================================
        # Generate answer
        # ====================================

        try:

            answer = generate_answer(
                question,
                context
            )

        except Exception as e:

            print(
                f"\n❌ LLM error: {e}"
            )

            continue

        # ====================================
        # IMPORTANT:
        # Calculate confidence AFTER answer
        # ====================================

        confidence = get_confidence(
            best_distance,
            answer
        )

        # ====================================
        # Answer
        # ====================================

        print("\n" + "-" * 60)

        print("Answer:")

        print(answer)

        # ====================================
        # Confidence
        # ====================================

        print(
            f"\nConfidence: {confidence}"
        )

        print(
            f"Best retrieval distance: "
            f"{best_distance:.4f}"
        )

        # ====================================
        # Sources
        # ====================================

        print("\n📚 Sources:")

        for i, (
            document,
            distance,
            metadata
        ) in enumerate(
            zip(
                documents,
                distances,
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

            print(
                f"\nSource {i}"
            )

            print(
                f"  File     : {source}"
            )

            print(
                f"  Page     : {page}"
            )

            print(
                f"  Chunk    : {chunk}"
            )

            print(
                f"  Distance : {distance:.4f}"
            )

        print("-" * 60)

        # ====================================
        # Save conversation
        # ====================================

        conversation_history.append(
            {
                "role": "user",
                "content": question
            }
        )

        conversation_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


# ============================================
# Run
# ============================================

if __name__ == "__main__":
    main()