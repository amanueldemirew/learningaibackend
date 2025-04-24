import json
import tempfile
import os
import logging
from typing import List, Optional
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    BackgroundTasks,
    Query,
)
from sqlmodel import Session, select, func
from app.core.deps import get_current_active_user
from app.db.session import get_session
from app.models.course import Course, Module, Unit, Content, File
from app.models.user import User
from app.schemas.course import (
    ContentCreate,
    ContentUpdate,
    ContentResponse,
    ContentGenerationOptions,
)
from app.schemas.common import PaginatedResponse, ErrorResponse
from app.services.llama_index import LlamaIndexService
from app.utils.extract import PDFTableOfContents
from app.utils.logger import get_logger
from fastapi.responses import StreamingResponse
from app.core.supabase import download_file_from_supabase

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/contents",
    response_model=ContentResponse,
    status_code=status.HTTP_201_CREATED,
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
async def create_content(
    content: ContentCreate,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    # Get unit first to check module_id
    unit = db.get(Unit, content.unit_id)
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
    # Only course owner can create content
    if course.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )

    # Create content
    db_content = Content(**content.dict())
    db.add(db_content)
    db.commit()
    db.refresh(db_content)
    return db_content


@router.post(
    "/contents/generate",
    response_model=ContentResponse,
    status_code=status.HTTP_201_CREATED,
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
async def generate_content(
    unit_id: int = Query(..., description="Unit ID"),
    options: Optional[ContentGenerationOptions] = None,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    logger.info(f"Starting content generation for unit_id: {unit_id}")
    try:
        # Get unit first to check module_id
        logger.info("Fetching unit from database...")
        unit = db.get(Unit, unit_id)
        if not unit:
            logger.error(f"Unit not found with ID: {unit_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Unit not found"
            )
        logger.info(f"Found unit: {unit.title}")

        # Verify module exists and belongs to the course
        logger.info("Fetching module from database...")
        module = db.get(Module, unit.module_id)
        if not module:
            logger.error(f"Module not found with ID: {unit.module_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Module not found"
            )
        logger.info(f"Found module: {module.title}")

        # Verify course exists and user has access
        logger.info("Fetching course from database...")
        course = db.get(Course, module.course_id)
        if not course:
            logger.error(f"Course not found with ID: {module.course_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Course not found"
            )
        logger.info(f"Found course: {course.title}")

        # Only course owner can create content
        if course.user_id != current_user.id:
            logger.error(
                f"User {current_user.id} does not have permission to generate content for course {course.id}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
            )

        # Extract text from PDF if available
        source_text = ""
        page_reference = ""

        if course.file_id:
            logger.info("Course has associated file, attempting to extract text...")
            file = db.get(File, course.file_id)
            if file and file.file_path:
                temp_file_path = None
                try:
                    # Download file from Supabase to a temporary location
                    logger.info(f"Downloading file from Supabase: {file.file_path}")
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as temp_file:
                        temp_file_path = temp_file.name
                        file_content = await download_file_from_supabase(file.file_path)
                        temp_file.write(file_content)

                    # Extract text from PDF using temporary file
                    logger.info("Extracting text from PDF...")
                    pdf_toc = PDFTableOfContents(temp_file_path)
                    source_text = pdf_toc.extract_text_from_pages(1, 5)
                    logger.info(f"Extracted {len(source_text)} characters from PDF")
                except Exception as e:
                    logger.error(f"Error processing PDF: {str(e)}")
                finally:
                    # Clean up temporary file
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.unlink(temp_file_path)
                            logger.info("Temporary file cleaned up")
                        except PermissionError:
                            logger.warning(
                                f"Could not delete temporary file: {temp_file_path}"
                            )

        # If no source text is available, use a default prompt
        if not source_text:
            logger.info("No source text available, using default prompt")
            source_text = f"Create educational content for the unit: {unit.title}. if you do not have much imformation related to this unit, just create content that is in {unit.description}"

        # Generate content using LlamaIndex with options
        logger.info("Initializing LlamaIndex service...")
        llama_service = LlamaIndexService(collection_name=f"course_{module.course_id}")
        logger.info("LlamaIndex service initialized")

        # Create a prompt based on the options
        logger.info("Building generation prompt...")
        prompt = f"Generate comprehensive educational content for the unit: {unit.title} with exact name. just create content that is in {unit.description}"
        prompt += """Create detailed, informative content that explains the topic clearly .  
                    try to use as much as possible of the source text and graphics (use url only for graphics) for advanced concepts or thing that are hard to understand. 
                    do not display long string in one line.
                    The content must answer the learning objective of the unit."""
        prompt += "Use markdown formatting for the content of GitHub Flavored Markdown (GFM) . never use html tags and use  $ for mathimatical expressions must be in latex "

        if options:
            logger.info("Adding custom options to prompt...")
            if options.custom_prompt:
                prompt += f"\nCustom instructions: {options.custom_prompt}\n"
            if options.difficulty_level:
                prompt += f"Difficulty level: {options.difficulty_level}\n"
            if options.include_examples:
                prompt += "Include practical examples.\n"
            if options.include_exercises:
                prompt += "Include practice exercises.and solution with step by step explanation separetly\n"
            if options.tone:
                prompt += f"Use a {options.tone} tone.\n"
            if options.target_audience:
                prompt += f"Target audience: {options.target_audience}\n"

        # Generate content
        logger.info("Generating content using LlamaIndex...")
        try:
            content = await llama_service.generate_content(source_text, prompt)
            logger.info("Content generated successfully")
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate content: {str(e)}",
            )

        # Create content record
        logger.info("Creating content record in database...")
        content_metadata = options.dict() if options else {}

        db_content = Content(
            unit_id=unit_id,
            title=unit.title,
            content_type=options.content_type if options else "text",
            content=content,
            order=1,
            is_ai_generated=True,
            ai_prompt=prompt,
            page_reference=page_reference,
            content_metadata=json.dumps(content_metadata),
        )

        db.add(db_content)
        db.commit()
        db.refresh(db_content)
        logger.info("Content record created successfully")

        return db_content

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in content generation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )


@router.post(
    "/contents/batch-generate",
    response_model=dict,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        404: {"model": ErrorResponse, "description": "Course or module not found"},
        403: {"model": ErrorResponse, "description": "Not enough permissions"},
    },
)
async def batch_generate_content(
    module_id: int = Query(..., description="Module ID"),
    options: Optional[ContentGenerationOptions] = None,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    # Get module first to check course_id
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
    if course.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )

    # Get all units in the module
    units = db.query(Unit).filter(Unit.module_id == module_id).all()
    if not units:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No units found in the specified module",
        )

    # Generate content directly for each unit
    generated_contents = []
    for unit in units:
        content = await generate_content(unit.id, options, None, db, current_user)
        generated_contents.append(content.id)

    return {
        "message": f"Content generated successfully for {len(units)} units",
        "content_ids": generated_contents,
    }


@router.delete(
    "/contents/clear-module",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ErrorResponse, "description": "Course or module not found"},
        403: {"model": ErrorResponse, "description": "Not enough permissions"},
    },
)
async def clear_module_contents(
    module_id: int = Query(..., description="Module ID"),
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    # Get module first to check course_id
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
    if course.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )

    # Get all units in the module
    units = db.query(Unit).filter(Unit.module_id == module_id).all()
    if not units:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No units found in the specified module",
        )

    # Delete all contents for all units in the module
    for unit in units:
        db.query(Content).filter(Content.unit_id == unit.id).delete()

    db.commit()
    return None


@router.get(
    "/contents/module",
    response_model=PaginatedResponse[ContentResponse],
    responses={
        404: {"model": ErrorResponse, "description": "Course or module not found"},
        403: {"model": ErrorResponse, "description": "Not enough permissions"},
    },
)
async def list_module_contents(
    module_id: int = Query(..., description="Module ID"),
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
    sort_by: str = Query("order", description="Field to sort by"),
    sort_order: str = Query("asc", description="Sort order (asc or desc)"),
):
    # Get module first to check course_id
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
    # Allow access if user is owner or course is published
    if course.user_id != current_user.id and not course.is_published:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )

    # Build query to get all units in the module
    units_query = select(Unit.id).where(Unit.module_id == module_id)
    unit_ids = [unit.id for unit in db.exec(units_query).all()]

    # Build query for contents
    query = select(Content).where(Content.unit_id.in_(unit_ids))

    # Apply sorting
    if sort_order.lower() == "desc":
        query = query.order_by(getattr(Content, sort_by).desc())
    else:
        query = query.order_by(getattr(Content, sort_by).asc())

    # Get total count
    count_query = (
        select(func.count()).select_from(Content).where(Content.unit_id.in_(unit_ids))
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
    "/contents/unit",
    response_model=PaginatedResponse[ContentResponse],
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Course, module or unit not found",
        },
        403: {"model": ErrorResponse, "description": "Not enough permissions"},
    },
)
async def list_contents(
    unit_id: int = Query(..., description="Unit ID"),
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
    sort_by: str = Query("order", description="Field to sort by"),
    sort_order: str = Query("asc", description="Sort order (asc or desc)"),
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
    # Allow access if user is owner or course is published
    if course.user_id != current_user.id and not course.is_published:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
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


@router.put(
    "/contents/{content_id}",
    response_model=ContentResponse,
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Course, module, unit or content not found",
        },
        403: {"model": ErrorResponse, "description": "Not enough permissions"},
        400: {
            "model": ErrorResponse,
            "description": "Content does not belong to the specified unit",
        },
    },
)
async def update_content(
    content_id: int,
    content_update: ContentUpdate,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    # Get content first to check unit_id
    content = db.get(Content, content_id)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Content not found"
        )

    # Get unit to check module_id
    unit = db.get(Unit, content.unit_id)
    if not unit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Unit not found"
        )

    # Get module to check course_id
    module = db.get(Module, unit.module_id)
    if not module:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Module not found"
        )

    # Get course to check permissions
    course = db.get(Course, module.course_id)
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Course not found"
        )
    # Only course owner can update content
    if course.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )

    # Update content
    for key, value in content_update.dict(exclude_unset=True).items():
        setattr(content, key, value)

    db.add(content)
    db.commit()
    db.refresh(content)
    return content


@router.delete(
    "/contents/{content_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Course, module, unit or content not found",
        },
        403: {"model": ErrorResponse, "description": "Not enough permissions"},
    },
)
async def delete_content(
    content_id: int,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    # Get content first to check unit_id
    content = db.get(Content, content_id)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Content not found"
        )

    # Get unit to check module_id
    unit = db.get(Unit, content.unit_id)
    if not unit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Unit not found"
        )

    # Get module to check course_id
    module = db.get(Module, unit.module_id)
    if not module:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Module not found"
        )

    # Get course to check permissions
    course = db.get(Course, module.course_id)
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Course not found"
        )
    # Only course owner can delete content
    if course.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )

    db.delete(content)
    db.commit()
    return None


@router.get(
    "/contents/{content_id}",
    response_model=ContentResponse,
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Course, module, unit or content not found",
        },
        403: {"model": ErrorResponse, "description": "Not enough permissions"},
    },
)
async def get_content(
    content_id: int,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    # Get content first to check unit_id
    content = db.get(Content, content_id)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Content not found"
        )

    # Get unit to check module_id
    unit = db.get(Unit, content.unit_id)
    if not unit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Unit not found"
        )

    # Get module to check course_id
    module = db.get(Module, unit.module_id)
    if not module:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Module not found"
        )

    # Get course to check permissions
    course = db.get(Course, module.course_id)
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Course not found"
        )
    # Allow access if user is owner or course is published
    if course.user_id != current_user.id and not course.is_published:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )

    return content


@router.post(
    "/contents/{content_id}/regenerate",
    response_model=ContentResponse,
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Course, module, unit or content not found",
        },
        403: {"model": ErrorResponse, "description": "Not enough permissions"},
    },
)
async def regenerate_content(
    content_id: int,
    options: ContentGenerationOptions,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    # Get content first to check unit_id
    content = db.get(Content, content_id)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Content not found"
        )

    # Get unit to check module_id
    unit = db.get(Unit, content.unit_id)
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
    # Only course owner can regenerate content
    if course.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )

    # Generate content using AI
    try:
        # Extract text from the unit's pages
        file = db.get(File, course.file_id)
        source_text = ""

        if file and file.file_path:
            # Download file from Supabase to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file_path = temp_file.name
                file_content = await download_file_from_supabase(file.file_path)
                temp_file.write(file_content)

                try:
                    # Extract text from PDF using temporary file
                    pdf_toc = PDFTableOfContents(temp_file_path)
                    source_text = pdf_toc.extract_text_from_pages(1, 5)
                finally:
                    # Clean up temporary file
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.unlink(temp_file_path)
                        except PermissionError:
                            # Log the error but continue execution
                            logger.warning(
                                f"Could not delete temporary file: {temp_file_path}"
                            )

        # Create a prompt based on the options
        prompt = (
            f"Generate comprehensive educational content for the unit: {unit.title}. "
        )
        prompt += (
            "Create detailed, informative content that explains the topic clearly. "
        )

        if options.custom_prompt:
            prompt += f"\nCustom instructions: {options.custom_prompt}\n"
        if options.difficulty_level:
            prompt += f"Difficulty level: {options.difficulty_level}\n"
        if options.include_examples:
            prompt += "Include practical examples.\n"
        if options.include_exercises:
            prompt += "Include practice exercises.\n"
        if options.tone:
            prompt += f"Use a {options.tone} tone.\n"
        if options.target_audience:
            prompt += f"Target audience: {options.target_audience}\n"

        # Add a fallback instruction to ensure content is generated even if source text is minimal
        prompt += "\nIf the provided source text is insufficient, generate comprehensive educational content based on your knowledge about the topic."

        # Generate content
        generated_content = await LlamaIndexService(
            collection_name=f"course_{course.id}"
        ).generate_content(source_text, prompt)

        # Update content metadata
        content_metadata = options.dict()

        # Update the content with the generated text
        content_update = ContentUpdate(
            content=generated_content, content_metadata=json.dumps(content_metadata)
        )
        updated_content = await update_content(
            content_id, content_update, db, current_user
        )
        return updated_content

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate content: {str(e)}",
        )


@router.post(
    "/contents/batch-generate-all",
    response_model=dict,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        404: {"model": ErrorResponse, "description": "Course not found"},
        403: {"model": ErrorResponse, "description": "Not enough permissions"},
    },
)
async def batch_generate_all_content(
    course_id: int = Query(..., description="Course ID"),
    options: Optional[ContentGenerationOptions] = None,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    # Verify course exists and user has access
    course = db.get(Course, course_id)
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Course not found"
        )
    if course.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )

    # Get all modules in the course
    modules = db.query(Module).filter(Module.course_id == course_id).all()
    if not modules:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No modules found in the specified course",
        )

    # Get all units in all modules
    all_units = []
    for module in modules:
        units = db.query(Unit).filter(Unit.module_id == module.id).all()
        all_units.extend(units)

    if not all_units:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No units found in any module of the specified course",
        )

    # Generate content directly for each unit
    generated_contents = []
    for unit in all_units:
        content = await generate_content(unit.id, options, None, None, db, current_user)
        generated_contents.append(content.id)

    return {
        "message": f"Content generated successfully for {len(all_units)} units across {len(modules)} modules",
        "content_ids": generated_contents,
    }
