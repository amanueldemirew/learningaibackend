from sqlmodel import SQLModel, create_engine, Session, select
from app.core.config import settings
from app.core.security import get_password_hash
from app.models.user import User

engine = create_engine(
    settings.SUPABASE_CONNECTION_STRING,
    echo=True
)

def init_db():
    SQLModel.metadata.create_all(engine)
    create_default_user()


def create_default_user():
    """Create a default user if none exists"""
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
            print("Default user created: user@example.com / password123")


def get_session():
    with Session(engine) as session:
        yield session
