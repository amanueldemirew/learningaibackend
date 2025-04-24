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
    SUPABASE_KEY: str
    SUPABASE_SERVICE_ROLE_KEY: str
    SUPABASE_POOLED_URL: str
    DIRECT_URL: str
    SUPABASE_CONNECTION_STRING: str

    # Chunk Settings
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 20
    EMBED_DIM: int = 768
    DEBUG: bool = False
    RELOAD: bool = True
    LOG_LEVEL: str = "INFO"

    # Gemini Model Settings
    GEMINI_MODEL: str = "gemini-pro"
    GEMINI_EMBEDDING_MODEL: str = "models/embedding-001"
    GROQ_MODEL: str = "llama3-70b-8192"

    # BACKEND_CORS_ORIGINS is a comma-separated list of origins
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
