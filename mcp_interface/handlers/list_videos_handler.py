"""Handler for listing videos in S3 storage."""

import os
import json
import logging
import boto3
from botocore.client import Config
from typing import Dict, Any, List
import mcp.types as types

from .base_handler import BaseHandler

logger = logging.getLogger(__name__)


class ListVideosHandler(BaseHandler):
    """Handler for listing available videos in S3 storage."""
    
    def __init__(self):
        super().__init__()
        # Create S3 client for MinIO
        self.s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv('S3_ENDPOINT_URL', 'http://minio:9000'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin123'),
            config=Config(signature_version='s3v4'),
            region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        )
        self.bucket_name = os.getenv('S3_BUCKET_NAME', 'video-assets')
    
    def get_tool_definition(self) -> types.Tool:
        """Get MCP tool definition for list_videos."""
        return types.Tool(
            name="list_videos",
            description="List all available videos in the S3 storage bucket",
            inputSchema={
                "type": "object",
                "properties": {
                    "prefix": {
                        "type": "string",
                        "description": "Optional prefix to filter videos (e.g., 'uploads/', 'test_videos/')",
                        "default": ""
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of videos to return (default: 100)",
                        "default": 100
                    }
                },
                "required": []
            }
        )
    
    def can_handle(self, tool_name: str) -> bool:
        """Check if this handler can handle the given tool name.
        
        Args:
            tool_name: Name of the tool to check.
            
        Returns:
            True if this handler can handle the tool, False otherwise.
        """
        return tool_name == "list_videos"
    
    async def handle(self, tool_name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
        """Handle the tool call.
        
        Args:
            tool_name: Name of the tool being called.
            arguments: Arguments passed to the tool.
            
        Returns:
            List of text content responses.
        """
        if tool_name != "list_videos":
            raise ValueError(f"Handler cannot handle tool: {tool_name}")
        
        result = await self.execute(**(arguments or {}))
        
        # Convert result to TextContent
        return [
            types.TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """List videos in S3 bucket.
        
        Args:
            prefix: Optional prefix to filter videos
            max_results: Maximum number of results to return
            
        Returns:
            Dict containing list of video files with their S3 URLs and metadata
        """
        prefix = kwargs.get('prefix', '')
        max_results = kwargs.get('max_results', 100)
        
        try:
            logger.info(f"Listing videos in bucket {self.bucket_name} with prefix '{prefix}'")
            
            # List objects in bucket
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_results
            )
            
            videos = []
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    # Check if it's a video file
                    if any(key.lower().endswith(ext) for ext in video_extensions):
                        videos.append({
                            'name': os.path.basename(key),
                            'video_path': f"s3://{self.bucket_name}/{key}",  # Use S3 URL as video_path
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'].isoformat() if hasattr(obj['LastModified'], 'isoformat') else str(obj['LastModified'])
                        })
            
            logger.info(f"Found {len(videos)} videos in bucket")
            
            return {
                'videos': videos,
                'total_count': len(videos),
                'bucket': self.bucket_name,
                'prefix': prefix
            }
            
        except Exception as e:
            logger.error(f"Error listing videos: {e}")
            return {
                'error': str(e),
                'videos': [],
                'total_count': 0,
                'bucket': self.bucket_name
            }