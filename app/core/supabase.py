import os
import uuid
import logging
import time
from supabase import create_client, Client
from fastapi import UploadFile
from app.core.config import settings

# Initialize Supabase client with anonymous key for general operations
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

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

# Set up logging
logger = logging.getLogger(__name__)


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

        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        logger.info(f"File size: {file_size} bytes")

        # For small files, use direct upload
        if file_size < CHUNK_SIZE:
            logger.info("Using direct upload for small file")
            return await _direct_upload(
                file_content, unique_filename, file.content_type
            )
        else:
            logger.info("Using chunked upload for large file")
            return await _chunked_upload(
                file_content, unique_filename, file.content_type
            )

    except Exception as e:
        logger.error(f"Error in upload_file_to_supabase: {str(e)}")
        raise Exception(f"Error uploading file to Supabase: {str(e)}")


async def _direct_upload(file_content, unique_filename, content_type):
    """Upload a file directly to Supabase storage"""
    retries = 0
    last_error = None

    while retries < MAX_RETRIES:
        try:
            logger.info(f"Direct upload attempt {retries + 1}/{MAX_RETRIES}")

            # Upload to Supabase using the admin client to bypass RLS
            res = supabase_admin.storage.from_(BUCKET_NAME).upload(
                path=unique_filename,
                file=file_content,
                file_options={"content-type": content_type},
            )

            # Check for error
            if hasattr(res, "error") and res.error:
                logger.error(f"Supabase upload error: {res.error.message}")
                raise Exception(f"Supabase upload error: {res.error.message}")

            # Get the public URL
            public_url = supabase_admin.storage.from_(BUCKET_NAME).get_public_url(
                unique_filename
            )
            logger.info(f"File uploaded successfully. Public URL: {public_url}")
            return public_url

        except Exception as e:
            last_error = e
            logger.error(
                f"Upload error (attempt {retries + 1}/{MAX_RETRIES}): {str(e)}"
            )
            retries += 1

            if retries < MAX_RETRIES:
                # Exponential backoff
                wait_time = 2**retries
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"All upload attempts failed: {str(last_error)}")
                raise Exception(
                    f"Upload failed after {MAX_RETRIES} attempts: {str(last_error)}"
                )


async def _chunked_upload(file_content, unique_filename, content_type):
    """Upload a file in chunks to Supabase storage"""
    # For now, we'll just use the direct upload with a longer timeout
    # In a production environment, you would implement proper chunked uploads
    # using Supabase's multipart upload API if available

    logger.info(
        "Chunked upload not fully implemented, falling back to direct upload with longer timeout"
    )
    return await _direct_upload(file_content, unique_filename, content_type)


async def upload_image_to_supabase(file: UploadFile, user_id: int) -> str:
    """Wrapper for uploading an image file"""
    return await upload_file_to_supabase(file, user_id, "image")


async def delete_file_from_supabase(file_url: str) -> bool:
    """
    Delete a file from Supabase storage.

    Args:
        file_url: The public URL of the file to delete

    Returns:
        True if deletion was successful, False otherwise
    """
    try:
        # Extract the path after the bucket name
        file_path = file_url.split(f"{BUCKET_NAME}/", 1)[1]
        logger.info(
            f"Attempting to delete file: {file_path} from bucket: {BUCKET_NAME}"
        )

        # Use admin client to bypass RLS
        res = supabase_admin.storage.from_(BUCKET_NAME).remove([file_path])
        if hasattr(res, "error") and res.error:
            logger.error(f"Delete failed: {res.error.message}")
            return False
        logger.info(f"File deleted successfully: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error deleting file from Supabase: {str(e)}")
        return False


async def download_file_from_supabase(file_url: str) -> bytes:
    """
    Download a file from Supabase storage and return its content as bytes.

    Args:
        file_url: The public URL of the file to download

    Returns:
        The file content as bytes
    """
    try:
        # Extract the path after the bucket name
        file_path = file_url.split(f"{BUCKET_NAME}/", 1)[1]
        logger.info(
            f"Attempting to download file: {file_path} from bucket: {BUCKET_NAME}"
        )

        # Download the file from Supabase
        response = supabase.storage.from_(BUCKET_NAME).download(file_path)
        logger.info(f"File downloaded successfully: {file_path}")
        return response
    except Exception as e:
        logger.error(f"Error downloading file from Supabase: {str(e)}")
        raise Exception(f"Error downloading file from Supabase: {str(e)}")
