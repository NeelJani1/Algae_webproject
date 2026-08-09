import os
import logging
from typing import Optional
from huggingface_hub import hf_hub_download

from dotenv import load_dotenv

logger = logging.getLogger("SeaDino")

# Load environment variables
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
load_dotenv(dotenv_path=env_path, override=True)

def get_file_path(file_path_or_name: str, hf_repo: Optional[str] = None) -> str:
    """
    Resolves the file path locally or downloads it from Hugging Face Hub using a secure token.
    """
    if os.path.exists(file_path_or_name):
        return file_path_or_name
    
    if hf_repo:
        filename = os.path.basename(file_path_or_name)
        logger.info(f"Attempting to fetch '{filename}' from Hugging Face ({hf_repo})...")
        
        hf_token = os.getenv("HF_TOKEN")
        
        if not hf_token:
            logger.warning("No HF_TOKEN found! Private repos will fail.")
        else:
            # Safely print just the first 4 and last 4 characters of the token to verify it loaded
            logger.info(f"Loaded HF_TOKEN: {hf_token[:4]}...{hf_token[-4:]}")
            
        try:
            return hf_hub_download(repo_id=hf_repo, filename=filename, token=hf_token)
        except Exception as e:
            logger.error(f"Failed to fetch '{filename}'. Error: {e}")
            return file_path_or_name
            
    return file_path_or_name