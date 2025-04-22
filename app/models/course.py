from typing import Optional, List, TYPE_CHECKING
from sqlmodel import Field, Relationship, SQLModel
from app.models.base import TimestampModel
# Remove the direct import of File to avoid circular import
# from app.models.file import File


class File(TimestampModel, table=True):
    __tablename__ = "file"

    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str = Field(index=True)
    file_path: str
    file_size: int  # Size in bytes
    file_type: str  # MIME type
    user_id: int = Field(foreign_key="user.id")
    is_public: bool = Field(default=False)
    description: Optional[str] = Field(default=None)

    courses: List["Course"] = Relationship(
        back_populates="file", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )


class Course(TimestampModel, table=True):
    __tablename__ = "course"

    id: Optional[int] = Field(default=None, primary_key=True)
    title: str = Field(index=True)
    description: Optional[str] = Field(default=None)
    thumbnail_url: Optional[str] = Field(default=None)
    is_published: bool = Field(default=False)
    user_id: int = Field(foreign_key="user.id")
    file_id: Optional[int] = Field(default=None, foreign_key="file.id")

    # Relationships
    modules: List["Module"] = Relationship(
        back_populates="course",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    toc: Optional["TableOfContents"] = Relationship(
        back_populates="course",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    file: Optional[File] = Relationship(
        back_populates="courses"
    )  # Direct reference since File is in same file


class Module(TimestampModel, table=True):
    """Model for course modules that contain units"""

    __tablename__ = "module"

    id: Optional[int] = Field(default=None, primary_key=True)
    title: str = Field(index=True)
    description: Optional[str] = Field(default=None)
    order: int = Field(default=0)
    course_id: int = Field(foreign_key="course.id")

    # Relationships
    course: Course = Relationship(back_populates="modules")
    units: List["Unit"] = Relationship(
        back_populates="module",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class Unit(TimestampModel, table=True):
    """Model for units within modules that contain content"""

    __tablename__ = "unit"

    id: Optional[int] = Field(default=None, primary_key=True)
    title: str = Field(index=True)
    description: Optional[str] = Field(default=None)
    order: int = Field(default=0)
    module_id: int = Field(foreign_key="module.id")

    # Relationships
    module: Module = Relationship(back_populates="units")
    contents: List["Content"] = Relationship(
        back_populates="unit", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )


class Content(TimestampModel, table=True):
    """Model for content within units, which can be AI-generated"""

    __tablename__ = "content"

    id: Optional[int] = Field(default=None, primary_key=True)
    title: str = Field(index=True)
    content_type: str = Field(default="text")  # text, video, quiz, etc.
    content: str  # The actual content or reference to content
    order: int = Field(default=0)
    unit_id: int = Field(foreign_key="unit.id")
    is_ai_generated: bool = Field(default=False)
    ai_prompt: Optional[str] = Field(
        default=None
    )  # The prompt used to generate content
    page_reference: Optional[str] = Field(
        default=None
    )  # Reference to specific pages in the PDF
    content_metadata: Optional[str] = Field(
        default=None
    )  # JSON string containing additional metadata

    # Relationships
    unit: Unit = Relationship(back_populates="contents")


class TableOfContents(TimestampModel, table=True):
    """Model for storing the table of contents for a course"""

    __tablename__ = "table_of_contents"

    id: Optional[int] = Field(default=None, primary_key=True)
    course_id: int = Field(foreign_key="course.id", unique=True)
    toc_data: str  # JSON string containing the TOC structure
    is_auto_generated: bool = Field(
        default=True
    )  # Whether the TOC was auto-generated from PDF

    # Relationships
    course: Course = Relationship(back_populates="toc")
