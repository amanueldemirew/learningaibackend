import os
import uuid
import logging
import time
from supabase import create_client, Client
from fastapi import UploadFile
from app.core.config import settings
import urllib.parse

# Set up logging
logger = logging.getLogger(__name__)

# Initialize Supabase client with service role key for general operations
supabase: Client = create_client(
    settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY
)

# Initialize Supabase admin client with service role key for storage operations
# This bypasses RLS policies
supabase_admin: Client = create_client(
    settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY
)

# Use a bucket that already exists in your Supabase project
BUCKET_NAME = "test3"
MAX_RETRIES = 3
UPLOAD_TIMEOUT = 60  # seconds
CHUNK_SIZE = 1024 * 1024  # 1MB chunks


async def upload_file_to_supabase(
    file: UploadFile, user_id: int, file_type: str = "pdf"
) -> str:
    """
    Upload a file to Supabase storage and return the public URL.

    Args:
        file: The uploaded file
        user_id: The ID of the user uploading the file
        file_type: The type of file (pdf, image, etc.)

    Returns:
        The public URL of the uploaded file
    """
    try:
        # Generate a unique filename with file extension
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{user_id}/{uuid.uuid4()}{file_ext}"

        logger.info(
            f"Attempting to upload file: {unique_filename} to bucket: {BUCKET_NAME}"
        )
        logger.info(f"File content type: {file.content_type}")

        # Read the file content
        file_content = await file.read()
        content_type = file.content_type or "application/octet-stream"

        # Upload the file
        if len(file_content) > CHUNK_SIZE:
            # Use chunked upload for large files
            url = await _chunked_upload(file_content, unique_filename, content_type)
        else:
            # Use direct upload for small files
            url = await _direct_upload(file_content, unique_filename, content_type)

        logger.info(f"File uploaded successfully: {url}")
        return url
    except Exception as e:
        logger.error(f"Error uploading file to Supabase: {str(e)}")
        raise


async def _direct_upload(file_content, unique_filename, content_type):
    """Upload a file directly to Supabase storage"""
    try:
        # Upload the file
        response = supabase_admin.storage.from_(BUCKET_NAME).upload(
            unique_filename, file_content, {"content-type": content_type}
        )

        # Get the public URL
        url = supabase_admin.storage.from_(BUCKET_NAME).get_public_url(unique_filename)
        return url
    except Exception as e:
        logger.error(f"Error in direct upload: {str(e)}")
        raise


async def _chunked_upload(file_content, unique_filename, content_type):
    """Upload a file in chunks to Supabase storage"""
    try:
        # Upload the file in chunks
        response = supabase_admin.storage.from_(BUCKET_NAME).upload(
            unique_filename, file_content, {"content-type": content_type}
        )

        # Get the public URL
        url = supabase_admin.storage.from_(BUCKET_NAME).get_public_url(unique_filename)
        return url
    except Exception as e:
        logger.error(f"Error in chunked upload: {str(e)}")
        raise


async def upload_image_to_supabase(file: UploadFile, user_id: int) -> str:
    """Upload an image to Supabase storage"""
    return await upload_file_to_supabase(file, user_id, "image")


async def delete_file_from_supabase(file_url: str) -> bool:
    """
    Delete a file from Supabase storage.

    Args:
        file_url: The public URL of the file to delete

    Returns:
        True if the file was deleted successfully, False otherwise
    """
    try:
        # Extract the path from the URL
        parsed_url = urllib.parse.urlparse(file_url)
        path = parsed_url.path

        # Remove the bucket name from the path
        if path.startswith(f"/storage/v1/object/public/{BUCKET_NAME}/"):
            path = path[len(f"/storage/v1/object/public/{BUCKET_NAME}/") :]

        # Delete the file
        response = supabase_admin.storage.from_(BUCKET_NAME).remove([path])
        logger.info(f"File deleted successfully: {file_url}")
        return True
    except Exception as e:
        logger.error(f"Error deleting file from Supabase: {str(e)}")
        return False


async def download_file_from_supabase(file_url: str) -> bytes:
    """
    Download a file from Supabase storage.

    Args:
        file_url: The public URL of the file to download

    Returns:
        The file content as bytes
    """
    try:
        # Extract the path from the URL
        parsed_url = urllib.parse.urlparse(file_url)
        path = parsed_url.path

        # Remove the bucket name from the path
        if path.startswith(f"/storage/v1/object/public/{BUCKET_NAME}/"):
            path = path[len(f"/storage/v1/object/public/{BUCKET_NAME}/") :]

        # Download the file
        response = supabase_admin.storage.from_(BUCKET_NAME).download(path)
        return response
    except Exception as e:
        logger.error(f"Error downloading file from Supabase: {str(e)}")
        raise
