from datetime import timedelta
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session, select
from app.core.config import settings
from app.core.security import create_access_token, get_password_hash, verify_password
from app.core.deps import get_current_active_user
from app.db.session import get_session
from app.models.user import User
from app.schemas.user import UserCreate, UserResponse, Token
from pydantic import BaseModel

router = APIRouter()


# Custom form class that uses email instead of username
class EmailPasswordRequestForm(OAuth2PasswordRequestForm):
    def __init__(
        self,
        email: str = Form(...),
        password: str = Form(...),
        grant_type: str = Form(None),
        scope: str = Form(""),
        client_id: str = Form(None),
        client_secret: str = Form(None),
    ):
        super().__init__(
            username=email,  # Pass email as username to parent class
            password=password,
            grant_type=grant_type,
            scope=scope,
            client_id=client_id,
            client_secret=client_secret,
        )
        self.email = email  # Store email separately for clarity


# New schema for JSON login
class LoginCredentials(BaseModel):
    email: str
    password: str


@router.post("/signup", response_model=UserResponse)
def signup(
    *,
    db: Session = Depends(get_session),
    user_in: UserCreate,
) -> Any:
    """
    Create new user.
    """
    # Check if user with this email exists
    user = db.exec(select(User).where(User.email == user_in.email)).first()
    if user:
        raise HTTPException(
            status_code=400,
            detail="The user with this email already exists in the system.",
        )

    # Check if user with this username exists
    user = db.exec(select(User).where(User.username == user_in.username)).first()
    if user:
        raise HTTPException(
            status_code=400,
            detail="The user with this username already exists in the system.",
        )

    # Create new user
    user = User(
        email=user_in.email,
        username=user_in.username,
        hashed_password=get_password_hash(user_in.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_session),
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests.
    This endpoint is used by the Swagger UI's Authorize button.

    The username field in the form should contain the user's email.
    """
    # Find user by email (which is passed as username in the form)
    user = db.exec(select(User).where(User.email == form_data.username)).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify password
    if not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user"
        )

    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(user.id, expires_delta=access_token_expires)

    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/login", response_model=Token)
def login(
    db: Session = Depends(get_session), credentials: LoginCredentials = None
) -> Any:
    """
    JSON login endpoint, get an access token for future requests.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing credentials",
        )

    # Find user by email
    user = db.exec(select(User).where(User.email == credentials.email)).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify password
    if not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user"
        )

    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(user.id, expires_delta=access_token_expires)

    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=UserResponse)
def read_users_me(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Get current user.
    """
    return current_user
