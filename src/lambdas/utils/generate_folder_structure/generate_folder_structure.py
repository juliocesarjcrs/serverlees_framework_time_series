
from fastapi import APIRouter, HTTPException, Request


from src.utils.utils import Utils

router = APIRouter()


@router.get("/utils/folder-structure")
async def generate_folder_structure(type_storage: str, path_base: str):
    utils = Utils()
    exclude_dirs = ['.git', '.serverless','buckets', '.pytest_cache', 'layers', 'node_modules', '__pycache__']
    utils.generate_folder_structure(path_base, exclude_dirs)

    return {'generate': 'ok'}