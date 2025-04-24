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


def create_db_engine(max_retries=5, retry_delay=5, use_vector_store=False):
    """Create database engine with retry logic

    Args:
        max_retries: Maximum number of connection attempts
        retry_delay: Delay between retries in seconds
        use_vector_store: Whether to use Supabase (True) or NeonDB (False)
    """
    # Determine which connection URL to use
    if use_vector_store:
        # For vector operations, use Supabase direct connection
        connection_url = settings.SUPABASE_CONNECTION_STRING
        connection_type = "supabase vector store"
    else:
        # For regular operations, use NeonDB
        connection_url = settings.DATABASE_URL
        connection_type = "neondb"

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

            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"Failed to connect to database after {max_retries} attempts"
                )
                raise
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
    # and doesn't have the pgbouncer parameter
    direct_url = (
        f"postgresql://{username}:{password}@{host}:5432/{database}?sslmode=require"
    )

    # Log the derived URL (with password masked)
    masked_direct_url = mask_connection_string(direct_url)
    logger.info(f"Derived direct connection URL: {masked_direct_url}")

    return direct_url


def mask_connection_string(connection_string):
    """Mask sensitive information in a connection string"""
    # Mask password in the connection string
    masked = re.sub(r":([^@]+)@", ":*****@", connection_string)
    return masked


# Create the engines
neondb_engine = create_db_engine(use_vector_store=False)
supabase_engine = create_db_engine(use_vector_store=True)


def init_db():
    try:
        # Initialize NeonDB tables
        SQLModel.metadata.create_all(neondb_engine)

        # Initialize Supabase vector extension
        with Session(supabase_engine) as session:
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            session.commit()
            logger.info("Vector extension created or already exists in Supabase")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        if not IS_DEVELOPMENT:
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
        with Session(neondb_engine) as session:
            # Check if any user exists
            users = session.exec(select(User)).all()
            if not users:
                # Create a default user
                default_user = User(
                    email="user@example.com",
                    username="testuser",
                    hashed_password=get_password_hash("12345"),
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


def get_session(use_vector_store=False):
    """Get a database session

    Args:
        use_vector_store: Whether to use Supabase (True) or NeonDB (False)
    """
    engine = supabase_engine if use_vector_store else neondb_engine
    with Session(engine) as session:
        yield session


def test_connections():
    """Test connections to both NeonDB and Supabase"""
    logger.info("Testing database connections...")

    # Test NeonDB connection
    try:
        with Session(neondb_engine) as session:
            session.execute(text("SELECT 1"))
            logger.info("NeonDB connection successful")
    except Exception as e:
        logger.error(f"NeonDB connection failed: {str(e)}")

    # Test Supabase connection
    try:
        with Session(supabase_engine) as session:
            session.execute(text("SELECT 1"))
            logger.info("Supabase connection successful")
    except Exception as e:
        logger.error(f"Supabase connection failed: {str(e)}")


def get_vector_engine():
    """Get a database engine specifically for vector operations using direct connection"""
    return create_db_engine(use_vector_store=False)
