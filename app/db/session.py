from sqlmodel import SQLModel, create_engine, Session, select
from app.core.config import settings
from app.core.security import get_password_hash
from app.models.user import User
import time
import logging
from sqlalchemy.exc import OperationalError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
                raise


# Create the engine
engine = create_db_engine()


def init_db():
    try:
        SQLModel.metadata.create_all(engine)
        create_default_user()
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
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
        raise


def get_session():
    with Session(engine) as session:
        yield session
