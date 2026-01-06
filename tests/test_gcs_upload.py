import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from questionbank.utils.gcs import get_gcs_client
from questionbank.config import config

from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gcs_upload():
    # Check if creds are set in env
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds:
        print(f"GOOGLE_APPLICATION_CREDENTIALS is set to: {creds}")
    else:
        print("WARNING: GOOGLE_APPLICATION_CREDENTIALS is NOT set in environment.")

    print(f"Testing GCS upload to bucket: {config.gcs.bucket_name}")
    
    if not config.gcs.bucket_name:
        print("ERROR: GCS_BUCKET_NAME not set.")
        return

    client = get_gcs_client()
    
    # Test upload bytes
    test_content = b"Hello GCS from QuestionBank!"
    blob_name = "test/hello_gcs.txt"
    
    print(f"Uploading bytes to {blob_name}...")
    url = client.upload_bytes(test_content, blob_name, content_type="text/plain")
    
    if url:
        print(f"SUCCESS: Uploaded bytes. URL: {url}")
    else:
        print("FAILURE: Upload bytes failed.")
        
    # Test upload file
    test_file = "test_gcs_file.txt"
    with open(test_file, "w") as f:
        f.write("Hello GCS file upload!")
        
    blob_name_file = "test/hello_gcs_file.txt"
    print(f"Uploading file to {blob_name_file}...")
    url = client.upload_file(test_file, blob_name_file)
    
    if url:
        print(f"SUCCESS: Uploaded file. URL: {url}")
    else:
        print("FAILURE: Upload file failed.")
        
    # Clean up local file
    if os.path.exists(test_file):
        os.remove(test_file)

if __name__ == "__main__":
    test_gcs_upload()
