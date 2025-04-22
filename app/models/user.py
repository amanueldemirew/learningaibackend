from typing import Optional
from sqlmodel import Field
from app.models.base import TimestampModel


class User(TimestampModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True)
    username: str = Field(unique=True, index=True)
    hashed_password: str
    is_active: bool = Field(default=True)
    is_superuser: bool = Field(default=False)
    profile_photo: Optional[str] = Field(
        default=None, description="Path to the user's profile photo"
    )
