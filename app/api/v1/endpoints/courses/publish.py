from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlmodel import Session
from app.core.deps import get_current_active_user
from app.db.session import get_session
from app.models.course import Course
from app.models.user import User
from app.schemas.course import CourseResponse
from app.schemas.common import ErrorResponse

router = APIRouter()


@router.put(
    "/publish",
    response_model=CourseResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Course not found"},
        403: {"model": ErrorResponse, "description": "Not enough permissions"},
        400: {
            "model": ErrorResponse,
            "description": "Course must have at least one module before publishing",
        },
    },
)
async def publish_course(
    course_id: int = Query(..., description="Course ID"),
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    # Get course and verify access
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

    # Verify course has required content
    if not course.modules:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Course must have at least one module before publishing",
        )

    # Publish course
    course.is_published = True
    db.commit()
    db.refresh(course)

    # Convert to CourseResponse with username
    course_dict = course.dict()
    course_dict["username"] = username
    return CourseResponse(**course_dict)


@router.put(
    "/unpublish",
    response_model=CourseResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Course not found"},
        403: {"model": ErrorResponse, "description": "Not enough permissions"},
    },
)
async def unpublish_course(
    course_id: int = Query(..., description="Course ID"),
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    # Get course and verify access
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

    # Unpublish course
    course.is_published = False
    db.commit()
    db.refresh(course)

    # Convert to CourseResponse with username
    course_dict = course.dict()
    course_dict["username"] = username
    return CourseResponse(**course_dict)


@router.get(
    "/published",
    response_model=List[CourseResponse],
    responses={
        403: {"model": ErrorResponse, "description": "Not enough permissions"},
    },
)
async def list_published_courses(
    skip: int = Query(0, description="Number of records to skip"),
    limit: int = Query(10, description="Number of records to return"),
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    # Get published courses with pagination
    results = (
        db.query(Course, User.username)
        .join(User, Course.user_id == User.id)
        .filter(Course.is_published == True)
        .offset(skip)
        .limit(limit)
        .all()
    )

    # Convert to CourseResponse objects
    courses = []
    for course, username in results:
        course_dict = course.dict()
        course_dict["username"] = username
        courses.append(CourseResponse(**course_dict))

    return courses


@router.get(
    "/published/{course_id}",
    response_model=CourseResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Course not found"},
        403: {"model": ErrorResponse, "description": "Course is not published"},
    },
)
async def get_published_course(
    course_id: int,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    # Get course and verify it exists and is published
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

    if not course.is_published:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Course is not published"
        )

    # Convert to CourseResponse with username
    course_dict = course.dict()
    course_dict["username"] = username
    return CourseResponse(**course_dict)


@router.get(
    "/published/user/{user_id}",
    response_model=List[CourseResponse],
    responses={
        404: {"model": ErrorResponse, "description": "User not found"},
    },
)
async def list_user_published_courses(
    user_id: int,
    skip: int = Query(0, description="Number of records to skip"),
    limit: int = Query(10, description="Number of records to return"),
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    # Get user and verify it exists
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # Get published courses by user with pagination
    results = (
        db.query(Course, User.username)
        .join(User, Course.user_id == User.id)
        .filter(Course.user_id == user_id, Course.is_published == True)
        .offset(skip)
        .limit(limit)
        .all()
    )

    # Convert to CourseResponse objects
    courses = []
    for course, username in results:
        course_dict = course.dict()
        course_dict["username"] = username
        courses.append(CourseResponse(**course_dict))

    return courses


@router.get(
    "/published/search",
    response_model=List[CourseResponse],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid search parameters"},
    },
)
async def search_published_courses(
    query: str = Query(..., description="Search query for title or description"),
    skip: int = Query(0, description="Number of records to skip"),
    limit: int = Query(10, description="Number of records to return"),
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    # Search published courses by title or description
    results = (
        db.query(Course, User.username)
        .join(User, Course.user_id == User.id)
        .filter(
            Course.is_published == True,
            (Course.title.ilike(f"%{query}%") | Course.description.ilike(f"%{query}%")),
        )
        .offset(skip)
        .limit(limit)
        .all()
    )

    # Convert to CourseResponse objects
    courses = []
    for course, username in results:
        course_dict = course.dict()
        course_dict["username"] = username
        courses.append(CourseResponse(**course_dict))

    return courses
