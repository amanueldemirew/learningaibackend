from sqlmodel import SQLModel, create_engine, Session, select
from app.core.config import settings
from app.core.security import get_password_hash
from app.models.user import User
import time
import logging
from sqlalchemy.exc import OperationalError
from sqlalchemy import text
import os
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flag to track if we're in a development environment
IS_DEVELOPMENT = os.getenv("ENVIRONMENT", "production").lower() == "development"


def create_db_engine(max_retries=5, retry_delay=5, use_pooled_connection=True):
    """Create database engine with retry logic

    Args:
        max_retries: Maximum number of connection attempts
        retry_delay: Delay between retries in seconds
        use_pooled_connection: Whether to use the pooled connection URL (True) or direct connection URL (False)
    """
    # Determine which connection URL to use
    if use_pooled_connection:
        # For regular operations, use the pooled connection
        if hasattr(settings, "SUPABASE_POOLED_URL") and settings.SUPABASE_POOLED_URL:
            connection_url = settings.SUPABASE_POOLED_URL
            connection_type = "supabase pooled"
        else:
            connection_url = settings.DATABASE_URL
            connection_type = "default pooled"
    else:
        # For vector operations, use the direct connection
        if hasattr(settings, "DIRECT_URL") and settings.DIRECT_URL:
            connection_url = settings.DIRECT_URL
            connection_type = "direct"
        else:
            # If DIRECT_URL is not available, try to derive it from DATABASE_URL
            connection_url = derive_direct_url(settings.DATABASE_URL)
            connection_type = "derived direct"

    logger.info(f"Attempting to connect to database using {connection_type} connection")
    logger.info(
        f"Connection parameters: max_retries={max_retries}, retry_delay={retry_delay}"
    )

    # Log connection details (without sensitive information)
    masked_url = mask_connection_string(connection_url)
    logger.info(f"Connecting to: {masked_url}")

    for attempt in range(max_retries):
        try:
            logger.info(f"Database connection attempt {attempt + 1}/{max_retries}")

            engine = create_engine(
                connection_url,
                echo=True,
                pool_pre_ping=True,  # Enable connection health checks
                pool_recycle=300,  # Recycle connections every 5 minutes
            )
            # Test the connection
            logger.info("Testing database connection...")
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                conn.commit()
            logger.info(
                f"Successfully connected to the database using {connection_type} connection"
            )
            return engine
        except OperationalError as e:
            logger.error(
                f"Database connection attempt {attempt + 1} failed with OperationalError: {str(e)}"
            )
            logger.error(f"Error details: {type(e).__name__}: {str(e)}")

            # Check if it's a network-related error
            if "Network is unreachable" in str(e):
                logger.error(
                    "Network connectivity issue detected. Please check your internet connection and firewall settings."
                )
                logger.error(
                    "If using a VPN, try disconnecting it. If behind a corporate firewall, check if the required port is allowed."
                )

            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"Failed to connect to database after {max_retries} attempts"
                )
                raise  # Re-raise the exception after all retries are exhausted
        except Exception as e:
            logger.error(
                f"Unexpected error connecting to database: {type(e).__name__}: {str(e)}"
            )
            logger.error(f"Error details: {str(e)}")
            raise


def derive_direct_url(pooled_url):
    """Derive a direct connection URL from a pooled connection URL"""
    logger.info("Deriving direct connection URL from pooled URL")

    # Extract components from the pooled URL
    match = re.match(r"postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/([^?]+)", pooled_url)
    if not match:
        logger.error("Failed to parse pooled connection URL")
        return pooled_url

    username, password, host, port, database = match.groups()

    # For Supabase, the direct connection typically uses port 5432 instead of 6543
    # and doesn't have the pgbouncer=true parameter
    direct_url = f"postgresql://{username}:{password}@{host}:5432/{database}"

    # Log the derived URL (with password masked)
    masked_direct_url = mask_connection_string(direct_url)
    logger.info(f"Derived direct connection URL: {masked_direct_url}")

    return direct_url


def mask_connection_string(connection_string):
    """Mask sensitive information in a connection string"""
    # Mask password in the connection string
    masked = re.sub(r":([^@]+)@", ":*****@", connection_string)
    return masked


# Create the engine with pooled connection by default
engine = create_db_engine()


def init_db():
    try:
        SQLModel.metadata.create_all(engine)
        create_vector_extension()
        create_default_user()
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        # Run diagnostics if there's a connection issue
        if "connection" in str(e).lower() or "network" in str(e).lower():
            logger.info("Running connection diagnostics due to connection error...")
            test_supabase_connection()

        if not IS_DEVELOPMENT:
            # In production, we'll continue even if initialization fails
            logger.warning(
                "Continuing application startup despite database initialization failure"
            )
        else:
            raise


def create_vector_extension():
    """Create the vector extension if it doesn't exist"""
    try:
        logger.info("Attempting to create vector extension...")
        # Use the vector-specific engine for vector operations
        vector_engine = get_vector_engine()
        with Session(vector_engine) as session:
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            session.commit()
            logger.info("Vector extension created or already exists")
    except Exception as e:
        logger.error(f"Error creating vector extension: {str(e)}")
        if not IS_DEVELOPMENT:
            # In production, we'll continue even if extension creation fails
            logger.warning(
                "Continuing application startup despite vector extension creation failure"
            )
        else:
            raise


def create_default_user():
    """Create a default user if none exists"""
    try:
        with Session(engine) as session:
            # Check if any user exists
            users = session.exec(select(User)).all()
            if not users:
                # Create a default user
                default_user = User(
                    email="user@example.com",
                    username="testuser",
                    hashed_password=get_password_hash("password123"),
                    is_active=True,
                )
                session.add(default_user)
                session.commit()
                logger.info("Default user created: user@example.com / password123")
    except Exception as e:
        logger.error(f"Error creating default user: {str(e)}")
        if not IS_DEVELOPMENT:
            # In production, we'll continue even if user creation fails
            logger.warning(
                "Continuing application startup despite user creation failure"
            )
        else:
            raise


def get_session():
    with Session(engine) as session:
        yield session


def test_supabase_connection():
    """Test connection to Supabase and provide detailed diagnostics"""
    import socket
    import requests
    from urllib.parse import urlparse

    logger.info("Running Supabase connection diagnostics...")

    # Test both pooled and direct connections
    for connection_type, url in [
        ("pooled", settings.DATABASE_URL),
        (
            "direct",
            getattr(settings, "DIRECT_URL", derive_direct_url(settings.DATABASE_URL)),
        ),
    ]:
        logger.info(f"Testing {connection_type} connection...")

        # Parse the database URL to extract host and port
        parsed_url = urlparse(url)
        host = parsed_url.hostname
        port = parsed_url.port or (6543 if connection_type == "pooled" else 5432)

        logger.info(f"Testing connection to Supabase host: {host} on port {port}")

        # Test basic network connectivity
        try:
            logger.info(f"Testing basic network connectivity to {host}...")
            socket.create_connection((host, port), timeout=5)
            logger.info(f"Basic network connectivity to {host}:{port} is working")
        except socket.timeout:
            logger.error(f"Connection to {host}:{port} timed out")
        except socket.gaierror as e:
            logger.error(f"DNS resolution failed for {host}: {str(e)}")
        except ConnectionRefusedError:
            logger.error(f"Connection to {host}:{port} was refused")
        except Exception as e:
            logger.error(
                f"Network connectivity test failed: {type(e).__name__}: {str(e)}"
            )

        # Test if the host is reachable via ping
        try:
            import subprocess

            logger.info(f"Testing ping to {host}...")
            result = subprocess.run(
                ["ping", "-c", "1", host], capture_output=True, text=True
            )
            if result.returncode == 0:
                logger.info(f"Ping to {host} successful")
            else:
                logger.error(f"Ping to {host} failed: {result.stderr}")
        except Exception as e:
            logger.error(f"Ping test failed: {type(e).__name__}: {str(e)}")

    # Test if we can connect to the Supabase API
    try:
        logger.info("Testing connection to Supabase API...")
        # Extract project reference from host (e.g., tlubcosubwqzpubyykvd from db.tlubcosubwqzpubyykvd.supabase.co)
        host = urlparse(settings.DATABASE_URL).hostname
        project_ref = host.split(".")[0] if host and "." in host else None
        if project_ref and project_ref.startswith("db."):
            project_ref = project_ref[3:]  # Remove 'db.' prefix

        if project_ref:
            api_url = f"https://{project_ref}.supabase.co/rest/v1/"
            response = requests.get(api_url, timeout=5)
            logger.info(
                f"Supabase API connection test: Status code {response.status_code}"
            )
        else:
            logger.warning(
                "Could not determine Supabase project reference for API test"
            )
    except Exception as e:
        logger.error(
            f"Supabase API connection test failed: {type(e).__name__}: {str(e)}"
        )

    logger.info("Supabase connection diagnostics completed")


def get_vector_engine():
    """Get a database engine specifically for vector operations using direct connection"""
    return create_db_engine(use_pooled_connection=False)
