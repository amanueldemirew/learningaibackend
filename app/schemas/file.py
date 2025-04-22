from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator



class FileBase(BaseModel):
    """
    Base file schema with common attributes.
    All files are associated with a specific user.
    """

    filename: str = Field(..., description="Name of the file")
    file_path: str = Field(..., description="Path where the file is stored")
    file_size: int = Field(..., description="Size of the file in bytes")
    file_type: str = Field(..., description="MIME type of the file")
    is_public: bool = Field(
        default=False, description="Whether the file is publicly accessible"
    )
    description: Optional[str] = Field(
        default=None, description="Optional description of the file"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the file was created",
    )
    updated_at: Optional[datetime] = Field(
        default=None, description="Timestamp when the file was last updated"
    )


class FileCreate(FileBase):
    """
    Schema for creating a new file.
    The user_id will be automatically set to the current user.
    """

    @validator("file_type")
    def validate_file_type(cls, v):
        if v != "application/pdf":
            raise ValueError("Only PDF files are allowed")
        return v


class FileUpload(BaseModel):
    """
    Schema for file upload requests.
    """

    description: Optional[str] = Field(
        default=None, description="Optional description of the file"
    )
    is_public: bool = Field(
        default=False, description="Whether the file should be publicly accessible"
    )


class FileUpdate(BaseModel):
    """
    Schema for updating a file.
    Only certain fields can be updated by the file owner.

    To update the file content, use the PUT endpoint with a new file upload.
    """

    filename: Optional[str] = Field(default=None, description="New name for the file")
    is_public: Optional[bool] = Field(
        default=None, description="Update file visibility"
    )
    description: Optional[str] = Field(
        default=None, description="Update file description"
    )


class FileResponse(FileBase):
    """
    Schema for file responses.
    Includes the file ID, user ID, and timestamp fields to identify ownership and track changes.
    """

    id: int = Field(..., description="Unique identifier of the file")
    user_id: int = Field(..., description="ID of the user who owns this file")

    class Config:
        from_attributes = True
