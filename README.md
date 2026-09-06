# 📄 Multi PDF RAG Bot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to ask questions across multiple PDF documents using LangChain, FAISS, Hugging Face Embeddings, OpenRouter, and Streamlit.

## 🚀 Features

* Chat with multiple PDF documents
* Semantic search using FAISS Vector Store
* Local embeddings with Hugging Face
* OpenRouter-powered LLM responses
* ChatGPT-style Streamlit interface
* Conversation history
* Fast document retrieval
* Source-aware document processing

---

## 🛠️ Tech Stack

* Python
* Streamlit
* LangChain
* OpenRouter
* Hugging Face Embeddings
* FAISS
* PyPDF
* Python Dotenv

---

## 📂 Project Structure

```text
MULTI-PDF-RAG/
│
├── app.py
├── requirements.txt
├── .env
│
└── RAG/
    ├── sample3.pdf
    └── sample4.pdf
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone <your-repository-url>
cd MULTI-PDF-RAG
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Create Environment File

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_api_key_here
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## 🧠 How It Works

1. Load multiple PDF documents.
2. Split documents into smaller chunks.
3. Generate embeddings using Hugging Face.
4. Store embeddings in a FAISS vector database.
5. Retrieve relevant chunks based on the user's question.
6. Send retrieved context to the LLM through OpenRouter.
7. Generate accurate answers grounded in the PDF content.

---

## 🔍 Retrieval Pipeline

```text
PDFs
  ↓
PyPDFLoader
  ↓
Text Splitter
  ↓
HuggingFace Embeddings
  ↓
FAISS Vector Store
  ↓
Retriever
  ↓
OpenRouter LLM
  ↓
Final Answer
```

---

## 📸 Demo

### Example Questions

* What information is available in the PDFs?
* Summarize the customer details.
* Which document contains specific information?
* What are the key insights from the uploaded PDFs?

---

## 🎯 Skills Demonstrated

* Retrieval-Augmented Generation (RAG)
* LangChain Development
* Vector Databases
* Semantic Search
* Prompt Engineering
* LLM Integration
* OpenRouter APIs
* Hugging Face Embeddings
* Streamlit Application Development
* Python Backend Development

---

## 🔮 Future Improvements

* PDF Upload Support
* Source Citations
* Chat Export
* Multi-user Sessions
* Advanced Search Filters
* Conversation Memory
* Hybrid Search (Keyword + Vector Search)

---

## 👨‍💻 Author

Sandeep Pilli

Diploma AI & ML Student | GenAI Enthusiast | Building AI Projects with LangChain, RAG, Streamlit, and OpenRouter.

---

⭐ If you found this project useful, consider giving it a star on GitHub.
