from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, validator
import secrets
from typing import List, Union
import os


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = ""  # Will be loaded from .env file
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    PROJECT_NAME: str = "FastAPI Backend"
    ALGORITHM: str = "HS256"

    # API Keys
    GOOGLE_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    GEMINI_API_KEY: str = ""

    # Supabase Configuration
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""  # Service role key with higher privileges
    SUPABASE_CONNECTION_STRING: str = ""  # Connection string for Supabase PostgreSQL
    SUPABASE_POOLED_URL: str = ""  # Pooled connection URL for regular operations
    DIRECT_URL: Optional[str] = (
        None  # Direct connection URL for vector operations and migrations
    )

    # Chunk Settings
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 32
    EMBED_DIM: int = 768
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    RELOAD: bool = os.getenv("RELOAD", "True").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Gemini Model Settings
    GEMINI_MODEL: str = "models/gemini-2.0-flash"
    GEMINI_EMBEDDING_MODEL: str = "models/embedding-001"

    # BACKEND_CORS_ORIGINS is a comma-separated list of origins
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # React default port
        "http://localhost:8000",  # FastAPI default port
        "http://localhost",
        "https://localhost",
        "https://localhost:3000",
        "http://localhost:5173",  # Vite default port
    ]

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # Database
    DATABASE_URL: str = "postgresql://neondb_owner:npg_0PSndjy7WsCz@ep-misty-bar-a467aet9-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require"

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
