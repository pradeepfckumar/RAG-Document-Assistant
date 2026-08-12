# 📚 RAG Document Assistant

A Retrieval-Augmented Generation (RAG) based document assistant that allows users to upload PDF documents and ask questions about their content.

The application extracts text from the uploaded PDF, splits it into smaller chunks, converts the chunks into embeddings, stores them in ChromaDB, retrieves the most relevant information for a question, and uses Groq's Llama 3.3 model to generate the final answer.

## 🚀 Features

- 📄 Upload PDF documents
- ✂️ Split documents into smaller chunks
- 🧠 Generate embeddings using Sentence Transformers
- 🗄️ Store embeddings in ChromaDB
- 🔍 Perform semantic search and retrieve relevant chunks
- 🤖 Generate answers using Groq Llama 3.3
- 💬 Ask questions about uploaded documents
- 🌐 Simple Streamlit web interface

## 🏗️ RAG Architecture

```text
PDF Document
     ↓
Text Extraction
     ↓
Document Chunking
     ↓
Sentence Transformer Embeddings
     ↓
ChromaDB Vector Store
     ↓
Semantic Retrieval
     ↓
Relevant Context
     ↓
Groq Llama 3.3
     ↓
Generated Answer
