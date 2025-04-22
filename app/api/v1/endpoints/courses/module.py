from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Response
from sqlmodel import Session, select, func
from app.core.deps import get_current_active_user
from app.db.session import get_session
from app.models.course import Course, Module
from app.models.user import User
from app.schemas.course import ModuleCreate, ModuleUpdate, ModuleResponse
from app.schemas.common import PaginatedResponse, ErrorResponse

router = APIRouter()


@router.post(
    "/modules",
    response_model=ModuleResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        404: {"model": ErrorResponse, "description": "Course not found"},
        403: {"model": ErrorResponse, "description": "Not enough permissions"},
    },
)
async def create_module(
    course_id: int = Query(..., description="ID of the course to create the module in"),
    module: ModuleCreate = None,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    # Verify course exists and user has access
    course = db.get(Course, course_id)
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Course not found"
        )
    # Only course owner can create modules
    if course.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )

    # Create module
    module_data = module.dict()
    module_data["course_id"] = course_id
    db_module = Module(**module_data)
    db.add(db_module)
    db.commit()
    db.refresh(db_module)
    return db_module


@router.get(
    "/modules",
    response_model=PaginatedResponse[ModuleResponse],
    responses={
        404: {"model": ErrorResponse, "description": "Course not found"},
        403: {"model": ErrorResponse, "description": "Not enough permissions"},
    },
)
async def list_modules(
    course_id: int,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
    sort_by: str = Query("order", description="Field to sort by"),
    sort_order: str = Query("asc", description="Sort order (asc or desc)"),
    search: Optional[str] = Query(
        None, description="Search term for title or description"
    ),
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

    # Build query
    query = select(Module).where(Module.course_id == course_id)

    # Apply search filter if provided
    if search:
        query = query.where(
            (Module.title.ilike(f"%{search}%"))
            | (Module.description.ilike(f"%{search}%"))
        )

    # Apply sorting
    if sort_order.lower() == "desc":
        query = query.order_by(getattr(Module, sort_by).desc())
    else:
        query = query.order_by(getattr(Module, sort_by).asc())

    # Get total count
    count_query = (
        select(func.count()).select_from(Module).where(Module.course_id == course_id)
    )
    if search:
        count_query = count_query.where(
            (Module.title.ilike(f"%{search}%"))
            | (Module.description.ilike(f"%{search}%"))
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


@router.put(
    "/modules/{module_id}",
    response_model=ModuleResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Course or module not found"},
        403: {"model": ErrorResponse, "description": "Not enough permissions"},
        400: {
            "model": ErrorResponse,
            "description": "Module does not belong to the specified course",
        },
    },
)
async def update_module(
    module_id: int,
    module_update: ModuleUpdate,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    # Get the module first to check course_id
    module = db.get(Module, module_id)
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
    # Only course owner can update modules
    if course.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )

    # Update module
    for key, value in module_update.dict(exclude_unset=True).items():
        setattr(module, key, value)

    db.add(module)
    db.commit()
    db.refresh(module)
    return module


@router.delete(
    "/modules/{module_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ErrorResponse, "description": "Course or module not found"},
        403: {"model": ErrorResponse, "description": "Not enough permissions"},
    },
)
async def delete_module(
    module_id: int,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    # Get the module first to check course_id
    module = db.get(Module, module_id)
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
    # Only course owner can delete modules
    if course.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )

    db.delete(module)
    db.commit()
    return None
