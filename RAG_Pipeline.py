import os
import uuid

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq


load_dotenv()


class RAG_Pipeline:
    def __init__(self):

        print("Loading embedding model...")

        # Embedding model
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print("Embedding model loaded.")

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        # ChromaDB
        self.client = chromadb.PersistentClient(
            path="data/vector_store"
        )

        self.collection = self.client.get_or_create_collection(
            name="documents"
        )

        print("ChromaDB initialized.")
        print(
            "Documents in collection:",
            self.collection.count()
        )

        # Groq
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in .env file"
            )

        self.llm = ChatGroq(
            groq_api_key=api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=1024
        )

        print("Groq LLM loaded.")
        print("RAG system ready.")


    def process_pdf(self, file_path):

        print("\nLoading PDF...")

        # Load PDF
        loader = PyPDFLoader(file_path)

        documents = loader.load()

        print(
            "Pages loaded:",
            len(documents)
        )

        # Split documents
        chunks = self.text_splitter.split_documents(
            documents
        )

        print(
            "Number of chunks:",
            len(chunks)
        )

        # Extract text
        texts = [
            chunk.page_content
            for chunk in chunks
        ]

        print("\nCreating embeddings...")

        # Generate embeddings
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True
        )

        # Generate IDs
        ids = [
            f"doc_{uuid.uuid4()}"
            for _ in chunks
        ]

        # Metadata
        metadata = [
            chunk.metadata
            for chunk in chunks
        ]

        # Store in ChromaDB
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadata
        )

        print(
            f"\n{len(chunks)} chunks added to ChromaDB."
        )

        return len(chunks)


    def ask(self, question, top_k=3):

        print("\nSearching document...")

        # Create query embedding
        query_embedding = self.embedding_model.encode(
            [question]
        )[0]

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=top_k
        )

        # Check results
        if (
            not results["documents"]
            or not results["documents"][0]
        ):
            return {
                "answer": (
                    "I could not find the answer "
                    "in the uploaded document."
                ),
                "sources": []
            }

        # Retrieved chunks
        documents = results["documents"][0]

        print(
            f"Retrieved {len(documents)} relevant chunks."
        )

        # Create context
        context = "\n\n".join(documents)

        # Prompt
        prompt = f"""
You are a document question-answering assistant.

Answer the question using ONLY the information
provided in the context below.

If the answer cannot be found in the context,
say:

"I could not find this information in the uploaded document."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""

        print("Generating answer...")

        # Generate answer
        response = self.llm.invoke(prompt)

        # Sources
        sources = results["metadatas"][0]

        return {
            "answer": response.content,
            "sources": sources
        }