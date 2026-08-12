import os
import streamlit as st
from RAG_Pipeline import RAG


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📚 RAG Document Assistant")

st.write(
    "Upload a PDF and ask questions about its content."
)


# --------------------------------------------------
# Initialize Session State
# --------------------------------------------------

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

if "rag" not in st.session_state:
    st.session_state.rag = RAG()


# --------------------------------------------------
# Upload PDF
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# --------------------------------------------------
# Process Document
# --------------------------------------------------

if uploaded_file is not None:

    if st.button("📄 Process Document"):

        os.makedirs("uploads", exist_ok=True)

        file_path = os.path.join(
            "uploads",
            uploaded_file.name
        )

        # Save uploaded PDF
        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        # Process PDF
        with st.spinner("Processing document..."):

            chunks = st.session_state.rag.process_pdf(
                file_path
            )

        st.session_state.document_processed = True

        st.success(
            f"Document processed successfully! "
            f"{chunks} chunks created."
        )


# --------------------------------------------------
# Question Answering
# --------------------------------------------------

if st.session_state.document_processed:

    st.divider()

    st.subheader("💬 Ask Questions")

    question = st.text_input(
        "Enter your question:"
    )

    if st.button("🔍 Ask Question"):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching document and generating answer..."
            ):

                result = st.session_state.rag.ask(
                    question
                )

            st.subheader("Answer")

            st.write(
                result["answer"]
            )

            # Sources
            if result["sources"]:

                st.subheader("📄 Sources")

                for source in result["sources"]:

                    st.write(source)