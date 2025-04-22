import os
import shutil
from fastapi import UploadFile


async def save_upload_file(upload_file: UploadFile, user_id: int) -> str:
    """
    Save an uploaded file to a user-specific folder and return the file path.

    Args:
        upload_file: The uploaded file
        user_id: The ID of the user uploading the file

    Returns:
        The path to the saved file
    """
    # Create user-specific upload directory
    folder = os.path.join("uploads", str(user_id))
    os.makedirs(folder, exist_ok=True)

    # Generate a unique filename
    file_path = os.path.join(folder, upload_file.filename)

    # Save the file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return file_path
