import os
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Form, UploadFile, File
from sqlmodel import Session, select
from app.core.deps import get_current_active_user
from app.db.session import get_session
from app.models.course import Course, File as FileModel
from app.models.user import User
from app.schemas.course import CourseResponse, CourseUpdate
from app.core.utils import save_upload_file
from app.core.supabase import upload_file_to_supabase, delete_file_from_supabase

router = APIRouter()


@router.post("/", response_model=CourseResponse, status_code=status.HTTP_201_CREATED)
async def create_course(
    title: str = Form(...),
    description: Optional[str] = Form(None),
    thumbnail_url: Optional[str] = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Create a new course with an uploaded PDF file."""
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported",
        )

    try:
        # Upload file to Supabase
        file_url = await upload_file_to_supabase(file, current_user.id)

        # Create file record
        db_file = FileModel(
            user_id=current_user.id,
            filename=file.filename,
            file_path=file_url,  # Store the Supabase URL
            file_type="application/pdf",
            file_size=file.size,  # Use the file size from the upload
        )
        db.add(db_file)
        db.commit()
        db.refresh(db_file)

        # Create course
        db_course = Course(
            user_id=current_user.id,
            file_id=db_file.id,
            title=title,
            description=description,
            thumbnail_url=thumbnail_url,
        )
        db.add(db_course)
        db.commit()
        db.refresh(db_course)

        # Convert to CourseResponse with username
        course_dict = db_course.dict()
        course_dict["username"] = current_user.username
        return CourseResponse(**course_dict)

    except Exception as e:
        # If there's an error, try to clean up the uploaded file
        if "file_url" in locals():
            await delete_file_from_supabase(file_url)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating course: {str(e)}",
        )


@router.post(
    "/by-file-id", response_model=CourseResponse, status_code=status.HTTP_201_CREATED
)
async def create_course_by_file_id(
    title: str = Form(...),
    description: Optional[str] = Form(None),
    thumbnail_url: Optional[str] = Form(None),
    file_id: int = Form(...),
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Create a new course using an existing file ID."""
    # Verify file exists and belongs to the user
    db_file = db.get(FileModel, file_id)
    if not db_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="File not found"
        )

    if db_file.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to use this file",
        )

    # Verify file is a PDF
    if not db_file.file_type == "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported",
        )

    # Create course
    db_course = Course(
        user_id=current_user.id,
        file_id=file_id,
        title=title,
        description=description,
        thumbnail_url=thumbnail_url,
    )
    db.add(db_course)
    db.commit()
    db.refresh(db_course)

    # Convert to CourseResponse with username
    course_dict = db_course.dict()
    course_dict["username"] = current_user.username
    return CourseResponse(**course_dict)


@router.get("/", response_model=List[CourseResponse])
async def list_courses(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Get a list of courses for the current user."""
    query = (
        select(Course, User.username)
        .join(User, Course.user_id == User.id)
        .where(Course.user_id == current_user.id)
        .offset(skip)
        .limit(limit)
    )
    results = db.exec(query).all()

    # Convert to CourseResponse objects
    courses = []
    for course, username in results:
        course_dict = course.dict()
        course_dict["username"] = username
        courses.append(CourseResponse(**course_dict))

    return courses


@router.get("/{course_id}", response_model=CourseResponse)
async def get_course(
    course_id: int,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Get a specific course by ID."""
    result = (
        db.query(Course, User.username)
        .join(User, Course.user_id == User.id)
        .filter(Course.id == course_id)
        .first()
    )

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Course not found"
        )

    course, username = result

    if course.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )

    # Convert to CourseResponse with username
    course_dict = course.dict()
    course_dict["username"] = username
    return CourseResponse(**course_dict)


@router.put("/{course_id}", response_model=CourseResponse)
async def update_course(
    course_id: int,
    course_update: CourseUpdate,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Update a course."""
    result = (
        db.query(Course, User.username)
        .join(User, Course.user_id == User.id)
        .filter(Course.id == course_id)
        .first()
    )

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Course not found"
        )

    course, username = result

    if course.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )

    # Update only provided fields
    update_data = course_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(course, field, value)

    db.commit()
    db.refresh(course)

    # Convert to CourseResponse with username
    course_dict = course.dict()
    course_dict["username"] = username
    return CourseResponse(**course_dict)


@router.delete("/{course_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_course(
    course_id: int,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Delete a course."""
    course = db.get(Course, course_id)
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Course not found"
        )

    if course.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )

    db.delete(course)
    db.commit()
    return None
