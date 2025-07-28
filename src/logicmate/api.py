# uvicorn logicmate.api:app --reload --port 8085

from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import FileResponse
import os

app = FastAPI(
    title="LogicMate API",
    description="API for LogicMateBot to handle image retrieval",
    version="1.0.0",
)
router = APIRouter()

BASE_MEDIA: str = os.path.abspath(path="media")


@router.get(path="/image-by-path/")
def get_image_by_path(path: str) -> FileResponse:
    full_path: str = os.path.abspath(path=path)

    if not full_path.startswith(BASE_MEDIA):
        raise HTTPException(status_code=400, detail="Invalid path")

    if not os.path.exists(path=full_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(path=full_path)


@router.get(path="/video-by-path/")
def get_video_by_path(path: str) -> FileResponse:
    full_path: str = os.path.abspath(path=path)

    if not full_path.startswith(BASE_MEDIA):
        raise HTTPException(status_code=400, detail="Invalid path")

    if not os.path.exists(path=full_path):
        raise HTTPException(status_code=404, detail="Video not found")

    return FileResponse(path=full_path, media_type="video/mp4")


app.include_router(router=router)
