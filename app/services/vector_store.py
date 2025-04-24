import os
import logging
from typing import List, Optional, Dict, Any
from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.supabase import SupabaseVectorStore
from supabase import create_client
from app.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

# Supabase credentials
supabase_url = settings.SUPABASE_URL
supabase_key = settings.SUPABASE_SERVICE_ROLE_KEY
client = create_client(supabase_url, supabase_key)

# Embedding model
embed_model = GoogleGenAIEmbedding(model_name=settings.GEMINI_EMBEDDING_MODEL)


class VectorStoreService:
    """Service for managing vector storage operations using Supabase"""

    def __init__(self, collection_name: str = "documents"):
        """
        Initialize the vector store service

        Args:
            collection_name: Name of the collection to use
        """
        self.collection_name = collection_name
        self.vector_store = None
        self.index = None
        self._init_vector_store()

    def _init_vector_store(self):
        """Initialize the Supabase vector store"""
        try:
            logger.info(
                f"Initializing Supabase vector store with collection: {self.collection_name}"
            )

            # Create vector store with Supabase
            self.vector_store = SupabaseVectorStore(
                client=client,
                table_name=self.collection_name,
                content_column="content",
                embedding_column="embedding",
                metadata_column="metadata",
            )

            # Load documents from directory
            documents = SimpleDirectoryReader("").load_data()

            # Create index
            self.index = VectorStoreIndex.from_documents(
                documents,
                vector_store=self.vector_store,
                embed_model=embed_model,
            )

            logger.info(
                f"Successfully initialized Supabase vector store for collection: {self.collection_name}"
            )
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def add_documents(self, documents: List[Document]) -> int:
        """
        Add documents to the vector store

        Args:
            documents: List of documents to add

        Returns:
            Number of documents added
        """
        try:
            logger.info(f"Adding {len(documents)} documents to vector store")

            # Add documents to the index
            for doc in documents:
                self.index.insert(doc)

            logger.info(
                f"Successfully added {len(documents)} documents to vector store"
            )
            return len(documents)
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
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
            logger.info(f"Querying vector store with: {query}")

            # Create a query engine
            query_engine = self.index.as_query_engine(similarity_top_k=top_k)

            # Query the index
            response = query_engine.query(query)

            logger.info("Successfully queried vector store")
            return str(response)
        except Exception as e:
            logger.error(f"Error querying vector store: {str(e)}")
            raise

    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Get a document by ID

        Args:
            doc_id: Document ID

        Returns:
            Document if found, None otherwise
        """
        try:
            logger.info(f"Getting document by ID: {doc_id}")

            # Get document from vector store
            doc = self.vector_store.get(doc_id)

            if doc:
                logger.info(f"Successfully retrieved document: {doc_id}")
                return doc
            else:
                logger.warning(f"Document not found: {doc_id}")
                return None
        except Exception as e:
            logger.error(f"Error getting document by ID: {str(e)}")
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
            result = self.vector_store.delete(doc_id)

            if result:
                logger.info(f"Successfully deleted document: {doc_id}")
                return True
            else:
                logger.warning(f"Document not found for deletion: {doc_id}")
                return False
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection

        Returns:
            Dictionary with collection statistics
        """
        try:
            logger.info("Getting collection statistics")

            # Return basic collection information
            stats = {
                "collection_name": self.collection_name,
                "vector_store_type": "supabase",
                "embedding_model": settings.GEMINI_EMBEDDING_MODEL,
            }

            logger.info("Successfully retrieved collection statistics")
            return stats
        except Exception as e:
            logger.error(f"Error getting collection statistics: {str(e)}")
            raise
