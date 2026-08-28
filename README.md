"RAG_AI_ASSISTANT"

Project details:
- The application is a Retrieval-Augmented Generation (RAG) knowledge assistant.
- It supports PDF, DOCX and TXT documents.
- Documents are loaded, split into chunks, converted into embeddings and stored in ChromaDB.
- User questions are converted into embeddings and relevant document chunks are retrieved.
- The retrieved context is passed to the language model to generate an answer.
- The UI is a ChatGPT-style Streamlit interface.
- The UI displays total documents, total pages, total chunks and average chunk size.
- Actions include Clear Conversation and Reindex Documents.
- Answers display confidence levels and document sources.
- Sources include file name, page number, chunk number and retrieval distance.
- Confidence levels are:
  Distance < 1.00 = High Confidence
  Distance < 1.50 = Medium Confidence
  Distance >= 1.50 = Low Confidence
- The application has been tested with empty PDF, corrupted PDF, very large PDF, question not in document, duplicate documents/chunks and unsupported file types.
- Sample documents include AI Notes, Company HR Policy, Employee Handbook, Leave Policy, empty PDF and corrupted PDF.
- The project contains app.py, ui.py, chatbot.py, chunker.py, config.py, document_loader.py, embeddings.py, logger.py, prompts.py, retriever.py, utils.py, requirements.txt and data/.
- Setup uses Python virtual environment and requirements.txt.
- The application is started using: streamlit run app.py
- Environment variables use HF_TOKEN. Never expose the real token in the README.
- Include a safe .env.example format:
  HF_TOKEN=your_huggingface_token_here

  ## Folder Structure

rag_Assistant/
│
├── app.py
├── ui.py
├── chatbot.py
├── chunker.py
├── config.py
├── document_loader.py
├── embeddings.py
├── logger.py
├── prompts.py
├── retriever.py
├── utils.py
│
├── requirements.txt
├── README.md
├── TEST_REPORT.md
├── .gitignore
├── .env.example
│
├── data/
│   ├── AI_Notes.txt
│   ├── Company_HR_Policy.txt
│   ├── Employee_Handbook.txt
│   ├── Leave_Policy.txt
│   ├── Mainreport.pdf
│   ├── Mainreport - Copy.pdf
│   ├── empty.pdf
│   └── corrupted.pdf
│
└── chroma_db/

## Setup Instructions

### Step 1: Clone the Repository

git clone https://github.com/Geethanprasath/RAG_AI_ASSISTANT.git

cd RAG_AI_ASSISTANT

### Step 2: Create Virtual Environment

python -m venv venv

### Step 3: Activate Virtual Environment

Windows:

venv\Scripts\activate

### Step 4: Install Required Packages

pip install -r requirements.txt

### Step 5: Configure Environment Variables

Create a `.env` file in the project directory.

Add your Hugging Face token:

HF_TOKEN=your_huggingface_token

### Step 6: Add Documents

Place PDF, DOCX and TXT documents inside the `data` folder.

### Step 7: Run the Application

streamlit run app.py

### Step 8: Open the Application

Open the following address in your browser:

http://localhost:8501

## Screenshots

### RAG Assistant Interface

The application provides a ChatGPT-style interface for asking questions about uploaded documents.

![RAG Assistant UI](screenshots/rag_assistant_ui.png)

### Question and Answer

The system displays the generated answer along with the confidence level.

![Question Answer](screenshots/question_answer.png)

### Sources

The application displays the retrieved document sources, including file name, page, chunk and retrieval distance.

![Sources](screenshots/sources.png)

### Document Statistics

The sidebar displays:

- Total Documents
- Total Pages
- Total Chunks
- Average Chunk Size

![Statistics](screenshots/statistics.png)

### Actions

The application provides:

- Clear Conversation
- Reindex Documents

![Actions](screenshots/actions.png)


## Known Limitations

1. Corrupted or invalid documents cannot be processed.

2. Empty documents may generate zero chunks.

3. Scanned PDFs may require OCR for accurate text extraction.

4. Very large documents can take more time to process and generate embeddings.

5. Duplicate documents may create duplicate chunks.

6. Answer quality depends on the quality of document text and retrieved chunks.

7. Questions unrelated to the uploaded documents may return a low-confidence response.

8. The current application supports PDF, DOCX and TXT document formats.

9. The application requires the configured embedding model and language model to be available.

10. The current confidence score is based on retrieval distance thresholds.


## Future Improvements

1. Add OCR support for scanned PDF documents.

2. Implement automatic duplicate document detection.

3. Add support for additional document formats.

4. Improve the confidence scoring mechanism.

5. Add conversation memory.

6. Add user authentication and access control.

7. Allow users to upload documents directly through the UI.

8. Add document deletion and document management features.

9. Improve source citation with better page-level navigation.

10. Improve retrieval accuracy using advanced retrieval techniques.

11. Add multilingual document and question support.

12. Deploy the RAG Assistant to a cloud platform.
