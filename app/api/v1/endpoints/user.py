from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import RedirectResponse
from sqlmodel import Session
from app.core.deps import get_current_active_user
from app.db.session import get_session
from app.models.user import User
from app.schemas.user import UserResponse, UserUpdate, PasswordChange
from app.core.security import get_password_hash, verify_password
from app.core.supabase import upload_image_to_supabase, delete_file_from_supabase
import os
import logging

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/me", response_model=UserResponse)
async def read_user_me(
    current_user: User = Depends(get_current_active_user),
):
    """Get current user"""
    return current_user


@router.put("/me", response_model=UserResponse)
async def update_user_me(
    user_update: UserUpdate,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Update current user"""
    update_data = user_update.dict(exclude_unset=True)

    # Handle password update separately
    if "password" in update_data:
        update_data["hashed_password"] = get_password_hash(update_data.pop("password"))

    for field, value in update_data.items():
        setattr(current_user, field, value)

    db.add(current_user)
    db.commit()
    db.refresh(current_user)
    return current_user


@router.post("/me/profile-photo", response_model=UserResponse)
async def upload_profile_photo(
    file: UploadFile = File(...),
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Upload a profile photo for the current user"""
    # Validate file type
    allowed_extensions = [".jpg", ".jpeg", ".png"]
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPG, JPEG, and PNG files are allowed for profile photos",
        )

    try:
        logger.info(
            f"Uploading profile photo for user {current_user.id}: {file.filename}"
        )
        logger.info(f"File content type: {file.content_type}")

        # Upload file to Supabase
        file_url = await upload_image_to_supabase(file, current_user.id)
        logger.info(f"File uploaded successfully. URL: {file_url}")

        # If user already has a profile photo, delete the old one from Supabase
        if current_user.profile_photo:
            logger.info(f"Deleting old profile photo: {current_user.profile_photo}")
            await delete_file_from_supabase(current_user.profile_photo)

        # Update user's profile photo URL
        current_user.profile_photo = file_url
        db.add(current_user)
        db.commit()
        db.refresh(current_user)
        logger.info(f"User profile photo updated successfully")

        return current_user
    except Exception as e:
        logger.error(f"Error uploading profile photo: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading profile photo: {str(e)}",
        )


@router.post("/me/change-password", response_model=UserResponse)
async def change_password(
    password_change: PasswordChange,
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Change the current user's password"""
    # Verify current password
    if not verify_password(
        password_change.current_password, current_user.hashed_password
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    # Update password
    current_user.hashed_password = get_password_hash(password_change.new_password)
    db.add(current_user)
    db.commit()
    db.refresh(current_user)

    return current_user


@router.get("/me/profile-photo")
async def get_profile_photo(
    current_user: User = Depends(get_current_active_user),
):
    """Get the profile photo for the current user"""
    if not current_user.profile_photo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile photo not found",
        )

    # Return a redirect to the Supabase URL
    return RedirectResponse(url=current_user.profile_photo)
