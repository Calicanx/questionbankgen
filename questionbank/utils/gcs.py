import os
import logging
from typing import Optional
from google.cloud import storage
from questionbank.config import config

logger = logging.getLogger(__name__)

class GCSClient:
    """Client for Google Cloud Storage."""
    
    def __init__(self):
        self.bucket_name = config.gcs.bucket_name
        self._client = None
        self._bucket = None
        
        if not self.bucket_name:
            logger.warning("GCS_BUCKET_NAME not set. GCS features will be disabled.")

    @property
    def client(self):
        if self._client is None:
            try:
                self._client = storage.Client()
            except Exception as e:
                logger.error(f"Failed to initialize GCS client: {e}")
                raise
        return self._client

    @property
    def bucket(self):
        if self._bucket is None and self.bucket_name:
            try:
                self._bucket = self.client.bucket(self.bucket_name)
            except Exception as e:
                logger.error(f"Failed to get GCS bucket {self.bucket_name}: {e}")
                raise
        return self._bucket

    def upload_file(self, file_path: str, destination_blob_name: str, content_type: Optional[str] = None) -> Optional[str]:
        """Uploads a file to the bucket."""
        if not self.bucket:
            return None
            
        try:
            # Auto-detect content type if not provided
            if not content_type:
                import mimetypes
                content_type, _ = mimetypes.guess_type(file_path)
            
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_filename(file_path, content_type=content_type)
            
            # Make public if uniform bucket-level access is not enabled (optional, usually better to use public URL format)
            # blob.make_public() 
            
            public_url = f"https://storage.googleapis.com/{self.bucket_name}/{destination_blob_name}"
            logger.info(f"Uploaded file to GCS: {public_url}")
            return public_url
        except Exception as e:
            logger.error(f"Failed to upload file to GCS: {e}")
            return None

    def upload_bytes(self, data: bytes, destination_blob_name: str, content_type: Optional[str] = None) -> Optional[str]:
        """Uploads bytes to the bucket."""
        if not self.bucket:
            return None
            
        try:
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_string(data, content_type=content_type)
            
            public_url = f"https://storage.googleapis.com/{self.bucket_name}/{destination_blob_name}"
            logger.info(f"Uploaded bytes to GCS: {public_url}")
            return public_url
        except Exception as e:
            logger.error(f"Failed to upload bytes to GCS: {e}")
            return None

# Singleton
_gcs_client = None

def get_gcs_client() -> GCSClient:
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = GCSClient()
    return _gcs_client
