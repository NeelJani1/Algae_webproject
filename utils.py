import os
import logging
from typing import Optional
from huggingface_hub import hf_hub_download

logger = logging.getLogger("SeaDino")

def get_file_path(file_path_or_name: str, hf_repo: Optional[str] = None) -> str:
    """
    Resolves the file path locally or downloads it from Hugging Face Hub.
    """
    if os.path.exists(file_path_or_name):
        return file_path_or_name
    
    if hf_repo:
        filename = os.path.basename(file_path_or_name)
        logger.info(f"Downloading '{filename}' from Hugging Face ({hf_repo})...")
        try:
            return hf_hub_download(repo_id=hf_repo, filename=filename)
        except Exception as e:
            logger.error(f"Failed to fetch '{filename}'. Error: {e}")
            return file_path_or_name
            
    return file_path_or_name