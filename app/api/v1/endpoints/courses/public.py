from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlmodel import Session, select, func, or_
from app.db.session import get_session
from app.models.course import Course, Module, Unit, Content
from app.models.user import User
from app.schemas.course import CourseResponse
from app.schemas.module import ModuleResponse
from app.schemas.unit import UnitResponse
from app.schemas.content import ContentResponse
from app.schemas.common import ErrorResponse, PaginatedResponse

router = APIRouter()


@router.get(
    "/public/courses",
    response_model=List[CourseResponse],
    responses={
        404: {"model": ErrorResponse, "description": "No published courses found"},
    },
)
async def list_public_courses(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
    db: Session = Depends(get_session),
):
    """Get a list of all published courses."""
    skip = (page - 1) * per_page
    courses = (
        db.query(Course, User.username)
        .join(User, Course.user_id == User.id)
        .filter(Course.is_published == True)
        .offset(skip)
        .limit(per_page)
        .all()
    )

    # Convert the results to CourseResponse objects
    result = []
    for course, username in courses:
        course_dict = course.dict()
        course_dict["username"] = username
        result.append(CourseResponse(**course_dict))

    return result


@router.get(
    "/public/courses/search",
    response_model=List[CourseResponse],
    responses={
        404: {"model": ErrorResponse, "description": "No published courses found"},
    },
)
async def search_public_courses(
    query: str = Query(..., description="Search query"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
    db: Session = Depends(get_session),
):
    """Search for published courses by title or description."""
    skip = (page - 1) * per_page
    courses = (
        db.query(Course, User.username)
        .join(User, Course.user_id == User.id)
        .filter(
            Course.is_published,
            or_(
                Course.title.ilike(f"%{query}%"),
                Course.description.ilike(f"%{query}%"),
            ),
        )
        .offset(skip)
        .limit(per_page)
        .all()
    )

    # Convert the results to CourseResponse objects
    result = []
    for course, username in courses:
        course_dict = course.dict()
        course_dict["username"] = username
        result.append(CourseResponse(**course_dict))

    return result


@router.get(
    "/public/courses/{course_id}",
    response_model=CourseResponse,
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Course not found or not published",
        },
    },
)
async def get_public_course(
    course_id: int,
    db: Session = Depends(get_session),
):
    """Get a specific published course by ID."""
    result = (
        db.query(Course, User.username)
        .join(User, Course.user_id == User.id)
        .filter(Course.id == course_id)
        .first()
    )

    if not result or not result[0].is_published:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Course not found or not published",
        )

    course, username = result
    course_dict = course.dict()
    course_dict["username"] = username

    return CourseResponse(**course_dict)


@router.get(
    "/public/courses/{course_id}/modules",
    response_model=PaginatedResponse[ModuleResponse],
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Course not found or not published",
        },
    },
)
async def list_public_modules(
    course_id: int,
    db: Session = Depends(get_session),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
    sort_by: str = Query("order", description="Field to sort by"),
    sort_order: str = Query("asc", description="Sort order (asc or desc)"),
):
    """Get a list of modules for a published course."""
    # Verify course exists and is published
    course = db.get(Course, course_id)
    if not course or not course.is_published:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Course not found or not published",
        )

    # Build query
    query = select(Module).where(Module.course_id == course_id)

    # Apply sorting
    if sort_order.lower() == "desc":
        query = query.order_by(getattr(Module, sort_by).desc())
    else:
        query = query.order_by(getattr(Module, sort_by).asc())

    # Get total count
    count_query = (
        select(func.count()).select_from(Module).where(Module.course_id == course_id)
    )
    total = db.exec(count_query).one()

    # Apply pagination
    query = query.offset((page - 1) * per_page).limit(per_page)

    # Execute query
    modules = db.exec(query).all()

    return PaginatedResponse(
        items=modules,
        total=total,
        page=page,
        per_page=per_page,
        total_pages=(total + per_page - 1) // per_page,
        has_next=page < (total + per_page - 1) // per_page,
        has_prev=page > 1,
    )


@router.get(
    "/public/modules/{module_id}/units",
    response_model=PaginatedResponse[UnitResponse],
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Module not found or course not published",
        },
    },
)
async def list_public_units(
    module_id: int,
    db: Session = Depends(get_session),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
    sort_by: str = Query("order", description="Field to sort by"),
    sort_order: str = Query("asc", description="Sort order (asc or desc)"),
):
    """Get a list of units for a module in a published course."""
    # Get module and verify course is published
    module = db.get(Module, module_id)
    if not module:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Module not found",
        )

    course = db.get(Course, module.course_id)
    if not course or not course.is_published:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Course not found or not published",
        )

    # Build query
    query = select(Unit).where(Unit.module_id == module_id)

    # Apply sorting
    if sort_order.lower() == "desc":
        query = query.order_by(getattr(Unit, sort_by).desc())
    else:
        query = query.order_by(getattr(Unit, sort_by).asc())

    # Get total count
    count_query = (
        select(func.count()).select_from(Unit).where(Unit.module_id == module_id)
    )
    total = db.exec(count_query).one()

    # Apply pagination
    query = query.offset((page - 1) * per_page).limit(per_page)

    # Execute query
    units = db.exec(query).all()

    return PaginatedResponse(
        items=units,
        total=total,
        page=page,
        per_page=per_page,
        total_pages=(total + per_page - 1) // per_page,
        has_next=page < (total + per_page - 1) // per_page,
        has_prev=page > 1,
    )


@router.get(
    "/public/units/{unit_id}/contents",
    response_model=PaginatedResponse[ContentResponse],
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Unit not found or course not published",
        },
    },
)
async def list_public_contents(
    unit_id: int,
    db: Session = Depends(get_session),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
    sort_by: str = Query("order", description="Field to sort by"),
    sort_order: str = Query("asc", description="Sort order (asc or desc)"),
):
    """Get a list of contents for a unit in a published course."""
    # Get unit and verify course is published
    unit = db.get(Unit, unit_id)
    if not unit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unit not found",
        )

    module = db.get(Module, unit.module_id)
    if not module:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Module not found",
        )

    course = db.get(Course, module.course_id)
    if not course or not course.is_published:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Course not found or not published",
        )

    # Build query
    query = select(Content).where(Content.unit_id == unit_id)

    # Apply sorting
    if sort_order.lower() == "desc":
        query = query.order_by(getattr(Content, sort_by).desc())
    else:
        query = query.order_by(getattr(Content, sort_by).asc())

    # Get total count
    count_query = (
        select(func.count()).select_from(Content).where(Content.unit_id == unit_id)
    )
    total = db.exec(count_query).one()

    # Apply pagination
    query = query.offset((page - 1) * per_page).limit(per_page)

    # Execute query
    contents = db.exec(query).all()

    return PaginatedResponse(
        items=contents,
        total=total,
        page=page,
        per_page=per_page,
        total_pages=(total + per_page - 1) // per_page,
        has_next=page < (total + per_page - 1) // per_page,
        has_prev=page > 1,
    )


@router.get(
    "/public/contents/{content_id}",
    response_model=ContentResponse,
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Content not found or course not published",
        },
    },
)
async def get_public_content(
    content_id: int,
    db: Session = Depends(get_session),
):
    """Get a specific content from a published course."""
    # Get content and verify course is published
    content = db.get(Content, content_id)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Content not found",
        )

    unit = db.get(Unit, content.unit_id)
    if not unit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unit not found",
        )

    module = db.get(Module, unit.module_id)
    if not module:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Module not found",
        )

    course = db.get(Course, module.course_id)
    if not course or not course.is_published:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Course not found or not published",
        )

    return content
