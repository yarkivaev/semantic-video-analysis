"""S3 utilities for handling S3 URLs in MCP handlers."""

import os
import tempfile
from urllib.parse import urlparse
import boto3
from botocore.client import Config
import logging

logger = logging.getLogger(__name__)


def is_s3_url(path: str) -> bool:
    """Check if a path is an S3 URL."""
    return path.startswith("s3://")


def parse_s3_url(s3_url: str) -> tuple[str, str]:
    """Parse S3 URL into bucket and key.
    
    Args:
        s3_url: S3 URL in format s3://bucket/key
        
    Returns:
        Tuple of (bucket, key)
    """
    parsed = urlparse(s3_url)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    return bucket, key


async def download_from_s3(s3_url: str) -> str:
    """Download file from S3 to a temporary location.
    
    Args:
        s3_url: S3 URL to download from
        
    Returns:
        Path to downloaded temporary file
    """
    bucket, key = parse_s3_url(s3_url)
    
    # Create S3 client for MinIO
    s3_client = boto3.client(
        's3',
        endpoint_url=os.getenv('S3_ENDPOINT_URL', 'http://minio:9000'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin123'),
        config=Config(signature_version='s3v4'),
        region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    )
    
    # Create temporary file
    suffix = os.path.splitext(key)[1]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = temp_file.name
    temp_file.close()
    
    try:
        logger.info(f"Downloading {s3_url} to {temp_path}")
        s3_client.download_file(bucket, key, temp_path)
        logger.info(f"Successfully downloaded {s3_url}")
        return temp_path
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise Exception(f"Failed to download from S3: {e}")


async def ensure_local_file(file_path: str) -> tuple[str, bool]:
    """Ensure file is available locally, downloading from S3 if needed.
    
    Args:
        file_path: Local path or S3 URL
        
    Returns:
        Tuple of (local_path, is_temp) where is_temp indicates if file should be cleaned up
    """
    if is_s3_url(file_path):
        local_path = await download_from_s3(file_path)
        return local_path, True
    else:
        # Already a local path
        return file_path, False