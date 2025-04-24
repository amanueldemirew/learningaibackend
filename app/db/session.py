from sqlmodel import SQLModel, create_engine, Session, select
from app.core.config import settings
from app.core.security import get_password_hash
from app.models.user import User
import time
import logging
from sqlalchemy.exc import OperationalError
from sqlalchemy import text
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flag to track if we're in a development environment
IS_DEVELOPMENT = os.getenv("ENVIRONMENT", "production").lower() == "development"


def create_db_engine(max_retries=5, retry_delay=5):
    """Create database engine with retry logic

    Args:
        max_retries: Maximum number of connection attempts
        retry_delay: Delay between retries in seconds
    """
    connection_url = settings.DATABASE_URL
    connection_type = "neondb"

    logger.info(f"Attempting to connect to database using {connection_type} connection")
    logger.info(
        f"Connection parameters: max_retries={max_retries}, retry_delay={retry_delay}"
    )

    # Log connection details
    logger.info(f"Connecting to: {connection_url}")

    for attempt in range(max_retries):
        try:
            logger.info(f"Database connection attempt {attempt + 1}/{max_retries}")

            engine = create_engine(
                connection_url,
                echo=True,
                pool_pre_ping=True,  # Enable connection health checks
                pool_recycle=300,  # Recycle connections every 5 minutes
                connect_args={"connect_timeout": 30},  # Increase connection timeout
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


# Create the engine
engine = create_db_engine()


def init_db():
    try:
        # Initialize tables
        SQLModel.metadata.create_all(engine)
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        if not IS_DEVELOPMENT:
            logger.warning(
                "Continuing application startup despite database initialization failure"
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


def get_session():
    """Get a database session"""
    with Session(engine) as session:
        yield session


def test_connections():
    """Test database connection"""
    logger.info("Testing database connection...")

    try:
        with Session(engine) as session:
            session.execute(text("SELECT 1"))
            logger.info("Database connection successful")
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
