from typing import List, Optional, Tuple

from pydantic import BaseModel, Field
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.video_stream import VideoStream

from logicmate.models.predictions.predictions.prediction import PredictionBase


class ImageModel(BaseModel):
    path: str
    category: Optional[str] = Field(default=None, description="Category of the image")
    predictions: Optional[List[PredictionBase]] = Field(
        default=None, description="List of predictions associated with the image"
    )


class Scene(BaseModel):
    scene_id: int
    start_timestamp: str
    end_timestamp: str
    images: List[ImageModel]


class Video(BaseModel):
    id: str
    duration: str
    categories: Optional[List[str]] = Field(
        default=None, description="List of categories associated with the video"
    )
    scenes: List[Scene]

    @classmethod
    def create_from_video_scenes(
        cls, video: VideoStream, scenes: List[Tuple[FrameTimecode, FrameTimecode]]
    ) -> "Video":
        """
        Create a Video instance from a VideoStream and a list of frames.

        Args:
            video (VideoStream): The video stream object.
            frames (List[Tuple[FrameTimecode, FrameTimecode]]): A list of tuples containing start and end timestamps.

        Returns:
            Video: An instance of the Video class.
        """
        # Will be 3 images per scene, but as we are starting from 1, we need to add 1
        images_per_scene = 4

        id = video.name
        duration = str(video.duration)
        scenes_list = []

        for i, (start_time, end_time) in enumerate(scenes, start=1):
            scene = Scene(
                scene_id=i,
                start_timestamp=str(start_time.get_timecode()),
                end_timestamp=str(end_time.get_timecode()),
                images=[
                    ImageModel(
                        path=f"{id}-Scene-{i:03d}-{j:02d}.png",
                    )
                    for j in range(1, images_per_scene)
                ],
            )
            scenes_list.append(scene)

        return cls(id=id, duration=duration, scenes=scenes_list)
