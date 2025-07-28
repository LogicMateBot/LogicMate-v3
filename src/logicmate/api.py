# uvicorn logicmate.api:app --reload --port 8000

from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import FileResponse
import os

app = FastAPI(
    title="LogicMate API",
    description="API for LogicMateBot to handle image retrieval",
    version="1.0.0",
)
router = APIRouter()

BASE_MEDIA = os.path.abspath("media")


@router.get("/image-by-path/")
def get_image_by_path(path: str):
    full_path = os.path.abspath(path)

    if not full_path.startswith(BASE_MEDIA):
        raise HTTPException(status_code=400, detail="Invalid path")

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(full_path)


app.include_router(router)
