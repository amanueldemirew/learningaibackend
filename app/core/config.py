from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyHttpUrl, validator
import secrets
from typing import List, Union
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    PROJECT_NAME: str = "FastAPI Backend"
    ALGORITHM: str = "HS256"

    # API Keys
    GOOGLE_API_KEY: str
    GROQ_API_KEY: str
    GEMINI_API_KEY: str

    # Database URLs
    DATABASE_URL: str
    SUPABASE_URL: str
    SUPABASE_SERVICE_ROLE_KEY: str
    SUPABASE_KEY: str
    SUPABASE_POSTGRES_CONNECTION_STRING: str

    # Chunk Settings
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 20
    EMBED_DIM: int = 768
    DEBUG: bool = False
    RELOAD: bool = True
    LOG_LEVEL: str = "INFO"

    # Gemini Model Settings
    GEMINI_MODEL: str = ""
    GEMINI_EMBEDDING_MODEL: str = ""
    GROQ_MODEL: str = ""

    # BACKEND_CORS_ORIGINS is a comma-separated list of origins
    BACKEND_CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8000, "https://learningai-eight.vercel.app/", "https://learningai-eight.vercel.app"

    @property
    def cors_origins(self) -> List[str]:
        """Get the CORS origins as a list."""
        if isinstance(self.BACKEND_CORS_ORIGINS, str):
            return [
                origin.strip()
                for origin in self.BACKEND_CORS_ORIGINS.split(",")
                if origin.strip()
            ]
        return self.BACKEND_CORS_ORIGINS

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )


settings = Settings()
