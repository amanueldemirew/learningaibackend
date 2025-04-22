from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlmodel import Session, select, func
from app.core.deps import get_current_active_user
from app.db.session import get_session
from app.models.course import Course, Module, Unit
from app.models.user import User
from app.schemas.course import UnitCreate, UnitUpdate, UnitResponse
from app.schemas.common import PaginatedResponse, ErrorResponse

router = APIRouter()


@router.post(
    "/units",
    response_model=UnitResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        404: {"model": ErrorResponse, "description": "Course or module not found"},
        403: {"model": ErrorResponse, "description": "Not enough permissions"},
        400: {
            "model": ErrorResponse,
            "description": "Module does not belong to the specified course",
        },
    },
)
async def create_unit(
    unit: UnitCreate,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    # Verify course exists and user has access
    module = db.get(Module, unit.module_id)
    if not module:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Module not found"
        )

    course = db.get(Course, module.course_id)
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Course not found"
        )
    # Only course owner can create units
    if course.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )

    # Create unit
    db_unit = Unit(**unit.dict())
    db.add(db_unit)
    db.commit()
    db.refresh(db_unit)
    return db_unit


@router.get(
    "/units",
    response_model=PaginatedResponse[UnitResponse],
    responses={
        404: {"model": ErrorResponse, "description": "Course or module not found"},
        403: {"model": ErrorResponse, "description": "Not enough permissions"},
    },
)
async def list_units(
    course_id: int = Query(..., description="Course ID"),
    module_id: int = Query(..., description="Module ID"),
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
    sort_by: str = Query("order", description="Field to sort by"),
    sort_order: str = Query("asc", description="Sort order (asc or desc)"),
):
    # Verify course exists and user has access
    course = db.get(Course, course_id)
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Course not found"
        )
    # Allow access if user is owner or course is published
    if course.user_id != current_user.id and not course.is_published:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )

    # Verify module exists and belongs to the course
    module = db.get(Module, module_id)
    if not module:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Module not found"
        )
    if module.course_id != course_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Module does not belong to the specified course",
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


@router.put(
    "/units/{unit_id}",
    response_model=UnitResponse,
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Course, module or unit not found",
        },
        403: {"model": ErrorResponse, "description": "Not enough permissions"},
        400: {
            "model": ErrorResponse,
            "description": "Unit does not belong to the specified module",
        },
    },
)
async def update_unit(
    unit_id: int,
    unit_update: UnitUpdate,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    # Get unit first to check module_id
    unit = db.get(Unit, unit_id)
    if not unit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Unit not found"
        )

    # Verify module exists and belongs to the course
    module = db.get(Module, unit.module_id)
    if not module:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Module not found"
        )

    # Verify course exists and user has access
    course = db.get(Course, module.course_id)
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Course not found"
        )
    # Only course owner can update units
    if course.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )

    # Update unit
    for key, value in unit_update.dict(exclude_unset=True).items():
        setattr(unit, key, value)

    db.add(unit)
    db.commit()
    db.refresh(unit)
    return unit


@router.delete(
    "/units/{unit_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Course, module or unit not found",
        },
        403: {"model": ErrorResponse, "description": "Not enough permissions"},
    },
)
async def delete_unit(
    unit_id: int,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    # Get unit first to check module_id
    unit = db.get(Unit, unit_id)
    if not unit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Unit not found"
        )

    # Verify module exists and belongs to the course
    module = db.get(Module, unit.module_id)
    if not module:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Module not found"
        )

    # Verify course exists and user has access
    course = db.get(Course, module.course_id)
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Course not found"
        )
    # Only course owner can delete units
    if course.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )

    db.delete(unit)
    db.commit()
    return None
