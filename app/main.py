from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import RedirectResponse
from app.core.config import settings
from app.api.v1.router import api_router
from app.db.session import init_db
from app.core.logger import logger
from app.core.middleware import TimeoutMiddleware

from llama_index.llms.google_genai import GoogleGenAI as Gemini
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding as GeminiEmbedding
from llama_index.core import Settings

import os
import time


CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "20"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")


def configure_settings():
    """Configure LlamaIndex settings"""

    # Reset any existing settings
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP
    Settings.llm = Gemini(model=GEMINI_MODEL, api_key=GEMINI_API_KEY)
    Settings.embed_model = GeminiEmbedding(model_name=GEMINI_EMBEDDING_MODEL)


# Create FastAPI app with increased file size limit
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    # Increase the maximum file size limit to 100MB
    max_request_size=100 * 1024 * 1024,  # 100MB in bytes
    debug=settings.DEBUG,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000", "https://learningaibackend.onrender.com/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Add the TimeoutMiddleware to the app
app.add_middleware(TimeoutMiddleware, timeout_seconds=120)  # 2 minutes timeout


# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=settings.PROJECT_NAME,
        version="1.0.0",
        description="API documentation with OAuth2 authentication",
        routes=app.routes,
    )

    # Add OAuth2 security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "OAuth2PasswordBearer": {
            "type": "oauth2",
            "flows": {
                "password": {
                    "tokenUrl": f"{settings.API_V1_STR}/auth/token",
                    "scopes": {},
                }
            },
        }
    }

    # Apply security globally
    openapi_schema["security"] = [{"OAuth2PasswordBearer": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Custom Swagger UI endpoint
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        title=f"{settings.PROJECT_NAME} - Swagger UI",
        oauth2_redirect_url="/docs/oauth2-redirect",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )


# OAuth2 redirect endpoint
@app.get("/docs/oauth2-redirect", include_in_schema=False)
async def swagger_ui_redirect():
    return RedirectResponse(url="/docs")


@app.on_event("startup")
async def startup_event():
    
    init_db()
    




@app.get("/")
async def root():
    return {"message": "Welcome to FastAPI Backend"}


