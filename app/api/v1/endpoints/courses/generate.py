import json
import tempfile
import os
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlmodel import Session
from app.core.deps import get_current_active_user
from app.db.session import get_session
from app.models.course import Course, Module, TableOfContents, Unit
from app.models.user import User
from app.schemas.toc import TOCResponse, TOCModule, TOCUnit, TOCMetadata
from app.utils.extract import PDFTableOfContents
from app.schemas.common import ErrorResponse
from datetime import datetime
from app.core.supabase import download_file_from_supabase
import logging

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/generate-toc",
    response_model=TOCResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Course not found"},
        403: {"model": ErrorResponse, "description": "Not enough permissions"},
    },
)
async def generate_course_toc(
    course_id: int = Query(..., description="Course ID"),
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
) -> TOCResponse:
    # Get the course
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Course not found"
        )

    # Check if user has access to the course
    if course.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )

    # Get the associated file
    if not course.file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No PDF file associated with this course",
        )

    try:
        # Download the file from Supabase
        file_content = await download_file_from_supabase(course.file.file_path)

        # Create a temporary file to store the PDF
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            # Extract TOC from PDF using the extract module
            toc_handler = PDFTableOfContents(temp_file_path)
            toc_data = toc_handler.convert_toc_to_json(temp_file_path)

            # Add course information to TOC data
            toc_data["course_id"] = course_id
            toc_data["course_title"] = course.title

            # Ensure metadata is present
            if "metadata" not in toc_data:
                toc_data["metadata"] = {
                    "generated_at": datetime.utcnow().isoformat(),
                    "source_file": course.file.file_path.split("/")[-1],
                }

            # Start a database transaction
            try:
                # Delete existing modules and units for this course
                # First, get all modules for this course
                existing_modules = (
                    db.query(Module).filter(Module.course_id == course_id).all()
                )

                # Delete units for each module
                for module in existing_modules:
                    db.query(Unit).filter(Unit.module_id == module.id).delete()

                # Then delete the modules
                db.query(Module).filter(Module.course_id == course_id).delete()

                # Create modules and units
                for module_data in toc_data["modules"]:
                    module = Module(
                        course_id=course_id,
                        title=module_data["title"],
                        description=module_data.get("description", ""),
                        order=module_data.get("order", 0),
                    )
                    db.add(module)
                    db.flush()  # Flush to get the module ID

                    for unit_data in module_data.get("units", []):
                        unit = Unit(
                            module_id=module.id,
                            title=unit_data["title"],
                            description=unit_data.get("description", ""),
                            order=unit_data.get("order", 0),
                            content_generated=unit_data.get("content_generated", False),
                        )
                        db.add(unit)

                # Commit to ensure all units get proper IDs
                db.commit()

                # Create or update TOC record
                toc_record = (
                    db.query(TableOfContents)
                    .filter(TableOfContents.course_id == course_id)
                    .first()
                )

                if toc_record:
                    toc_record.toc_data = json.dumps(toc_data)
                else:
                    toc_record = TableOfContents(
                        course_id=course_id,
                        toc_data=json.dumps(toc_data),
                        is_auto_generated=True,
                    )
                    db.add(toc_record)

                db.commit()

                # Get the newly created modules to access their IDs
                modules = db.query(Module).filter(Module.course_id == course_id).all()

                # Create a mapping of module titles to module IDs
                module_id_map = {module.title: module.id for module in modules}

                # Get all units for these modules
                module_ids = [module.id for module in modules]
                units = db.query(Unit).filter(Unit.module_id.in_(module_ids)).all()

                # Create a mapping of (module_id, unit_title) to unit_id
                unit_id_map = {(unit.module_id, unit.title): unit.id for unit in units}

                # Create the response
                response = TOCResponse(
                    course_id=course_id,
                    course_title=course.title,
                    modules=[
                        TOCModule(
                            id=module_id_map[module_data["title"]],
                            title=module_data["title"],
                            description=module_data.get("description", ""),
                            order=module_data.get("order", 0),
                            units=[
                                TOCUnit(
                                    id=unit_id_map.get(
                                        (
                                            module_id_map[module_data["title"]],
                                            unit["title"],
                                        ),
                                        0,
                                    ),
                                    title=unit["title"],
                                    description=unit.get("description", ""),
                                    order=unit.get("order", 0),
                                    content_generated=unit.get(
                                        "content_generated", False
                                    ),
                                )
                                for unit in module_data.get("units", [])
                            ],
                        )
                        for module_data in toc_data["modules"]
                    ],
                    metadata=TOCMetadata(
                        generated_at=toc_data["metadata"]["generated_at"],
                        source_file=toc_data["metadata"]["source_file"],
                    ),
                    is_complete=False,
                )

                return response

            except Exception as e:
                db.rollback()
                logger.error(f"Error generating TOC: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error generating TOC: {str(e)}",
                )

        finally:
            # Clean up the temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except PermissionError:
                    # Log the error but continue execution
                    logger.warning(f"Could not delete temporary file: {temp_file_path}")

    except Exception as e:
        db.rollback()
        logger.error(f"Error extracting table of contents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting table of contents: {str(e)}",
        )


@router.delete(
    "/clear-toc",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ErrorResponse, "description": "Course not found"},
        403: {"model": ErrorResponse, "description": "Not enough permissions"},
    },
)
async def clear_course_toc(
    course_id: int = Query(..., description="Course ID"),
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    # Get the course
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Course not found"
        )

    # Check if user has access to the course
    if course.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )

    try:
        # Delete all units associated with modules in this course
        modules = db.query(Module).filter(Module.course_id == course_id).all()
        for module in modules:
            db.query(Unit).filter(Unit.module_id == module.id).delete()

        # Delete all modules
        db.query(Module).filter(Module.course_id == course_id).delete()

        # Delete TOC record if exists
        db.query(TableOfContents).filter(
            TableOfContents.course_id == course_id
        ).delete()

        db.commit()
        return None

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing table of contents: {str(e)}",
        )
