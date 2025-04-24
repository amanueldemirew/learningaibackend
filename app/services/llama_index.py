import os
import json
import tempfile
from dotenv import load_dotenv
import requests
from io import BytesIO
import logging
import urllib.parse
import time
from typing import List, Optional, Dict, Any

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    get_response_synthesizer,
)
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core.schema import Document
from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.groq import Groq
from app.core.supabase import supabase, supabase_admin, BUCKET_NAME
from app.core.config import settings
from app.services.vector_store import VectorStoreService

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants from .env with defaults
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1024))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 20))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")

# Get the Supabase connection string from settings
SUPABASE_CONNECTION_STRING = settings.SUPABASE_URL

logger.info("Initializing LlamaIndex service with configuration:")
logger.info(f"CHUNK_SIZE: {CHUNK_SIZE}")
logger.info(f"CHUNK_OVERLAP: {CHUNK_OVERLAP}")
logger.info(f"GEMINI_MODEL: {GEMINI_MODEL}")
logger.info(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
logger.info(f"GROQ_MODEL: {GROQ_MODEL}")
logger.info(
    f"Supabase connection string configured: {'Yes' if SUPABASE_CONNECTION_STRING else 'No'}"
)

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY is not configured in environment variables")
    raise ValueError("GROQ_API_KEY is required in your environment")

if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY is not configured in environment variables")
    raise ValueError("GEMINI_API_KEY is required in your environment")

try:
    logger.info("Initializing Groq LLM...")
    llm = Groq(model=GROQ_MODEL, api_key=GROQ_API_KEY)
    logger.info("Groq LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Groq LLM: {str(e)}")
    raise


def configure_llama_settings():
    """Configure LlamaIndex settings"""
    # Reset any existing settings
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP

    # Configure LLM
    if GEMINI_API_KEY:
        Settings.llm = GoogleGenAI(model=GEMINI_MODEL, api_key=GEMINI_API_KEY)
    elif GROQ_API_KEY:
        Settings.llm = Groq(model=GROQ_MODEL, api_key=GROQ_API_KEY)
    else:
        logger.warning("No LLM API key found. Using default LLM.")

    # Configure embedding model
    if GEMINI_API_KEY:
        Settings.embed_model = GoogleGenAIEmbedding(model_name=EMBEDDING_MODEL)
    else:
        logger.warning("No embedding API key found. Using default embedding model.")


class LlamaIndexService:
    """Service for managing LlamaIndex operations"""

    def __init__(self, collection_name: str = "default"):
        """
        Initialize the LlamaIndex service

        Args:
            collection_name: Name of the collection to use
        """
        self.collection_name = collection_name
        self.vector_store = VectorStoreService(collection_name)

    def process_file(self, file_path: str) -> int:
        """
        Process a file and add it to the vector store

        Args:
            file_path: Path to the file to process

        Returns:
            Number of documents added
        """
        try:
            logger.info(f"Processing file: {file_path}")

            # Read the file
            documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

            # Add documents to vector store
            num_docs = self.vector_store.add_documents(documents)

            logger.info(f"Successfully processed file: {file_path}")
            return num_docs
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise

    def process_directory(self, directory_path: str) -> int:
        """
        Process all files in a directory and add them to the vector store

        Args:
            directory_path: Path to the directory to process

        Returns:
            Number of documents added
        """
        try:
            logger.info(f"Processing directory: {directory_path}")

            # Read all files in the directory
            documents = SimpleDirectoryReader(directory_path).load_data()

            # Add documents to vector store
            num_docs = self.vector_store.add_documents(documents)

            logger.info(f"Successfully processed directory: {directory_path}")
            return num_docs
        except Exception as e:
            logger.error(f"Error processing directory: {str(e)}")
            raise

    def query(self, query: str, top_k: int = 5) -> str:
        """
        Query the vector store

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            Response from the query
        """
        try:
            logger.info(f"Querying with: {query}")

            # Query the vector store
            response = self.vector_store.query(query, top_k)

            logger.info("Successfully queried vector store")
            return response
        except Exception as e:
            logger.error(f"Error querying vector store: {str(e)}")
            raise

    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get a document by ID

        Args:
            doc_id: Document ID

        Returns:
            Document if found, None otherwise
        """
        try:
            logger.info(f"Getting document: {doc_id}")

            # Get document from vector store
            doc = self.vector_store.get_document_by_id(doc_id)

            logger.info(f"Successfully retrieved document: {doc_id}")
            return doc
        except Exception as e:
            logger.error(f"Error getting document: {str(e)}")
            raise

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by ID

        Args:
            doc_id: Document ID

        Returns:
            True if document was deleted, False otherwise
        """
        try:
            logger.info(f"Deleting document: {doc_id}")

            # Delete document from vector store
            result = self.vector_store.delete_document(doc_id)

            logger.info(f"Successfully deleted document: {doc_id}")
            return result
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection

        Returns:
            Dictionary with collection statistics
        """
        try:
            logger.info("Getting collection statistics")

            # Get collection stats
            stats = self.vector_store.get_collection_stats()

            logger.info("Successfully retrieved collection statistics")
            return stats
        except Exception as e:
            logger.error(f"Error getting collection statistics: {str(e)}")
            raise
