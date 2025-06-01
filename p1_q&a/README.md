# 📄 Multi-PDF Question Answering App (LangChain + FAISS + Cassandra)

This project is a **PDF-based Question Answering (Q&A)** system where you can upload multiple PDF files and ask questions in natural language. It uses **LangChain**, **FAISS**, **HuggingFace**, and **Cassandra (via DataStax Astra DB)** to provide accurate answers from the documents.

---

## 🚀 Features

- ✅ Upload and process multiple PDFs
- 📚 Extract and chunk document content
- 🧠 Embed using `sentence-transformers`
- 🔍 Semantic search using **FAISS** or **Cassandra Vector Store**
- 🤖 Use HuggingFace LLM (e.g. `Mistral-7B-Instruct`)
- 🗨️ Natural Language Question Answering
- 💾 Persist data in vector DB (Cassandra or FAISS)

---

## 🛠️ Tech Stack

| Component      | Technology                          |
| -------------- | ----------------------------------- |
| 🧠 LLM         | HuggingFace (`mistralai/Mistral-7B-Instruct`) |
| 📚 Embedding   | `sentence-transformers/all-MiniLM-L6-v2` |
| 🧾 Parsing     | `PyPDFLoader` from LangChain         |
| 🔗 Framework   | `LangChain`                         |
| 📦 Vector DB   | `FAISS` or `Cassandra` via Astra DB |
| 🖥️ Frontend    | `Streamlit`                         |

---
