from sqlmodel import SQLModel, create_engine, Session, select
from app.core.config import settings
from app.core.security import get_password_hash
from app.models.user import User
import time
import logging
from sqlalchemy.exc import OperationalError
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flag to track if we're in a development environment
IS_DEVELOPMENT = os.getenv("ENVIRONMENT", "production").lower() == "development"


# Create engine with connection retry logic
def create_db_engine(max_retries=5, retry_delay=5):
    """Create database engine with retry logic"""
    for attempt in range(max_retries):
        try:
            engine = create_engine(
                settings.SUPABASE_CONNECTION_STRING,
                echo=True,
                pool_pre_ping=True,  # Enable connection health checks
                pool_recycle=300,  # Recycle connections every 5 minutes
            )
            # Test the connection
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            logger.info("Successfully connected to the database")
            return engine
        except OperationalError as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Database connection attempt {attempt + 1} failed: {str(e)}"
                )
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"Failed to connect to database after {max_retries} attempts: {str(e)}"
                )
                if IS_DEVELOPMENT:
                    # In development, use SQLite as fallback
                    logger.warning(
                        "Using SQLite as fallback database in development mode"
                    )
                    return create_sqlite_engine()
                else:
                    # In production, we'll use a delayed initialization approach
                    logger.warning(
                        "Using delayed database initialization in production mode"
                    )
                    return create_delayed_engine()
        except Exception as e:
            logger.error(f"Unexpected error connecting to database: {str(e)}")
            if IS_DEVELOPMENT:
                return create_sqlite_engine()
            else:
                return create_delayed_engine()


def create_sqlite_engine():
    """Create a SQLite engine for development fallback"""
    sqlite_url = "sqlite:///./sql_app.db"
    logger.info(f"Creating SQLite engine with URL: {sqlite_url}")
    return create_engine(sqlite_url, connect_args={"check_same_thread": False})


def create_delayed_engine():
    """Create a dummy engine that will be replaced later when connection is possible"""
    logger.info("Creating delayed initialization engine")
    # Create a dummy SQLite engine that will be replaced later
    dummy_engine = create_sqlite_engine()

    # Start a background thread to try connecting to the real database
    import threading

    def try_connect_later():
        time.sleep(10)  # Wait 10 seconds before trying
        try:
            real_engine = create_engine(
                settings.SUPABASE_CONNECTION_STRING,
                echo=True,
                pool_pre_ping=True,
                pool_recycle=300,
            )
            # Test the connection
            with real_engine.connect() as conn:
                conn.execute("SELECT 1")
            logger.info(
                "Successfully connected to the real database in background thread"
            )
            # Replace the global engine with the real one
            global engine
            engine = real_engine
        except Exception as e:
            logger.error(f"Background connection attempt failed: {str(e)}")

    # Start the background thread
    threading.Thread(target=try_connect_later, daemon=True).start()

    return dummy_engine


# Create the engine
engine = create_db_engine()


def init_db():
    try:
        SQLModel.metadata.create_all(engine)
        create_default_user()
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        if not IS_DEVELOPMENT:
            # In production, we'll continue even if initialization fails
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
