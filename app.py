import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import os

# Streamlit UI
st.title("Document Chatbot with RAG (Free Version)")

# File uploader (for local files)
uploaded_files = st.file_uploader("Upload multiple documents (PDF, DOCX, TXT)", accept_multiple_files=True)

# Initialize variables
docs = []

# Load and parse documents
for uploaded_file in uploaded_files:
    file_path = uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    if file_path.endswith(".pdf"):
        loader = UnstructuredPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        st.warning(f"Unsupported file type: {file_path}")
        continue

    docs.extend(loader.load())
    os.remove(file_path)

if docs:
    # Embed documents using Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever()

    # Simple text-based QA
    def simple_qa(query):
        docs = retriever.get_relevant_documents(query)
        answers = [doc.page_content for doc in docs]
        return "\n".join(answers) if answers else "No relevant information found."

    # Chat interface
    st.subheader("Ask questions about your documents")
    user_query = st.text_input("Your question:")

    if user_query:
        response = simple_qa(user_query)
        st.text_area("Response:", response)

# Run with: streamlit run app.py
