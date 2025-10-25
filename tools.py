# tools.py
import os
from crewai.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings # <-- 1. Import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

@tool
def pinecone_search(query: str) -> str:
    """
    A powerful search tool that queries a Pinecone vector index
    of academic research papers to find the most relevant information.
    """
    # 2. Create embeddings for the query using the local model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Initialize the Pinecone Vector Store
    vectorstore = PineconeVectorStore(
        index_name="academic-research",
        embedding=embeddings
    )

    # Perform the search
    docs = vectorstore.similarity_search(query, k=5)

    # Format the results
    return "\n---\n".join([doc.page_content for doc in docs])