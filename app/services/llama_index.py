import os
import json
import tempfile
from dotenv import load_dotenv
import requests
from io import BytesIO
import logging
import urllib.parse

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
from llama_index.core.schema import Document
from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.groq import Groq
from app.core.supabase import supabase, supabase_admin, BUCKET_NAME
from app.core.config import settings

# Set up logging
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

# Get the connection string from settings
SUPABASE_CONNECTION_STRING = settings.SUPABASE_CONNECTION_STRING



logger.info("Initializing LlamaIndex service with Supabase connection")

llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is required in your environment")


# Configure global settings once
def configure_llama_settings():
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP
    Settings.llm = llm
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name=EMBEDDING_MODEL, api_key=GEMINI_API_KEY
    )


class LlamaIndexService:
    def __init__(self, output_dir="output", course_id=None):
        """Initialize vector store and optional settings

        Args:
            output_dir: Base directory for storing vector indices
            course_id: Optional course ID to create a course-specific vector store
        """
        configure_llama_settings()
        self.output_dir = output_dir
        self.course_id = course_id

        # Initialize the vector store
        self._init_vector_store()
        self.vector_index = None

    def _init_vector_store(self):
        """Set up Supabase vector store"""
        # Use course-specific collection name if course_id is provided
        collection_name = (
            f"course_{self.course_id}" if self.course_id else "global_collection"
        )

        logger.info(
            f"Initializing Supabase vector store with collection: {collection_name}"
        )

        try:
            # Create the vector store
            self.vector_store = SupabaseVectorStore(
                postgres_connection_string=SUPABASE_CONNECTION_STRING,
                collection_name=collection_name,
            )

            # Create storage context
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            logger.info("Successfully initialized Supabase vector store")
        except Exception as e:
            logger.error(f"Error initializing Supabase vector store: {str(e)}")
            raise

    async def _download_from_supabase(self, file_url):
        """Download a file from Supabase and return its local path"""
        try:
            # Extract the file path from the URL
            file_path = file_url.split(f"{BUCKET_NAME}/")[1]

            # Download the file from Supabase
            response = supabase.storage.from_(BUCKET_NAME).download(file_path)

            # Create a temporary file
            temp_file_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(file_path)[1]
                ) as temp_file:
                    temp_file.write(response)
                    temp_file_path = temp_file.name
                    return temp_file_path
            except Exception as e:
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                    except PermissionError:
                        logger.error(
                            f"Could not delete temporary file: {temp_file_path}"
                        )
                raise e
        except Exception as e:
            logger.error(f"Error downloading file from Supabase: {str(e)}")
            raise

    async def process_pdf(self, pdf_path, start_page=None, end_page=None):
        """Load a PDF and build or update the vector index"""
        # Check if pdf_path is a URL (from Supabase)
        if pdf_path.startswith("http"):
            # Download the file from Supabase
            local_pdf_path = await self._download_from_supabase(pdf_path)
        else:
            local_pdf_path = pdf_path

        try:
            reader = PyMuPDFReader(start_page=start_page, end_page=end_page)
            docs = reader.load_data(file=local_pdf_path)
            if not docs:
                raise ValueError("No pages loaded from PDF")

            # Try to load existing index
            try:
                self.vector_index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store
                )
                logger.info("Loaded existing index.")
                self.vector_index.insert_nodes([Document(text=d.text) for d in docs])
                logger.info(f"Inserted {len(docs)} new pages into existing index.")
            except Exception as e:
                logger.warning(f"No existing index found or error loading: {str(e)}")
                logger.info("Building new index.")
                self.vector_index = VectorStoreIndex.from_documents(
                    docs, storage_context=self.storage_context
                )
                logger.info("Built new index.")

            # If we downloaded the file, clean up the temporary file
            if pdf_path.startswith("http"):
                os.unlink(local_pdf_path)

            return {"status": "ok", "pages_indexed": len(docs)}
        except Exception as e:
            # Clean up the temporary file in case of error
            if pdf_path.startswith("http") and os.path.exists(local_pdf_path):
                os.unlink(local_pdf_path)
            raise e

    async def query_index(self, query: str) -> str:
        """Query the stored vector index"""
        if self.vector_index is None:
            try:
                self.vector_index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store
                )
                logger.info("Loaded existing index for querying.")
            except Exception as e:
                logger.error(f"Error loading index: {str(e)}")
                raise RuntimeError(
                    "Vector index not initialized and couldn't be loaded. Process a PDF first."
                )

        retriever = self.vector_index.as_retriever(similarity_top_k=3)
        synth = get_response_synthesizer(llm=Settings.llm)
        engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=synth)
        response = engine.query(query)
        return response.response

    async def generate_content(self, text: str, prompt: str) -> str:
        """Index raw text (as a Document) and query with a prompt"""
        doc = Document(text=text)
        temp_idx = VectorStoreIndex.from_documents([doc])

        retriever = temp_idx.as_retriever(similarity_top_k=3)
        synth = get_response_synthesizer(llm=Settings.llm)
        engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=synth)
        response = engine.query(prompt)
        return response.response

    async def generate_summary(self, text: str) -> str:
        """Generate concise summary of text"""
        prompt = "Create a concise summary of the following content. Focus on the main points and key concepts. Keep the summary clear and informative."
        return await self.generate_content(text, prompt)
