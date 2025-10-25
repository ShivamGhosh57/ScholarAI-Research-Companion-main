# ingest.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # <-- 1. Import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

# Constants
PINECONE_INDEX_NAME = "academic-research"

def ingest_data():
    """
    Loads PDFs, splits them into chunks, creates embeddings,
    and upserts them to a Pinecone index.
    """
    print("Starting data ingestion...")

    # 1. Load documents from the data directory
    print("Loading documents...")
    loader = PyPDFDirectoryLoader("data/")
    documents = loader.load()
    if not documents:
        print("No documents found in the 'data' directory. Aborting.")
        return

    # 2. Split documents into chunks
    print(f"Splitting {len(documents)} documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"Created {len(docs)} chunks.")

    # 3. Create embeddings using a free, local model from Hugging Face
    print("Creating embeddings with a local Hugging Face model...")
    # This model is downloaded automatically and runs on your machine
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Upsert to Pinecone
    print(f"Upserting chunks to Pinecone index '{PINECONE_INDEX_NAME}'...")
    PineconeVectorStore.from_documents(
        docs,
        embeddings,
        index_name=PINECONE_INDEX_NAME
    )
    print("Data ingestion complete!")

if __name__ == "__main__":
    ingest_data()