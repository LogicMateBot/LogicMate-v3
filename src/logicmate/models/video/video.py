from typing import List, Optional, Tuple

from pydantic import BaseModel, Field
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.video_stream import VideoStream
from uuid import uuid4

from logicmate.models.predictions.predictions.prediction import PredictionBase


class ImageModel(BaseModel):
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the image",
    )
    path: str
    categories: Optional[List[str]] = Field(
        default=None, description="Category of the image"
    )
    explanation: Optional[str] = Field(
        default=None, description="Explanation of the image"
    )
    predictions: Optional[List[PredictionBase]] = Field(
        default=None, description="List of predictions associated with the image"
    )


class Scene(BaseModel):
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the scene",
    )
    start_timestamp: str
    end_timestamp: str
    categories: Optional[List[str]] = Field(
        default=None, description="Category of the scene"
    )
    images: List[ImageModel]
    explanation: Optional[str] = Field(
        default=None, description="Explanation of the scene"
    )


class Approach(BaseModel):
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the approach",
    )
    title: Optional[str] = Field(default=None, description="Title of the approach")
    description: Optional[str] = Field(
        default=None, description="General description of the approach"
    )
    originalCode: Optional[str] = Field(
        default=None, description="Original implementation code"
    )
    originalCodeExplanation: Optional[str] = Field(
        default=None, description="Explanation of the original implementation code"
    )
    newCode: Optional[str] = Field(
        default=None, description="Improved implementation code"
    )
    newCodeExplanation: Optional[str] = Field(
        default=None, description="Explanation of the improved implementation code"
    )


class Exercise(BaseModel):
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the approach",
    )
    title: Optional[str] = Field(default=None, description="Title of the exercise")
    description: Optional[str] = Field(
        default=None, description="Problem statement for the exercise"
    )
    solution: Optional[str] = Field(
        default=None, description="Code solution to the exercise"
    )


class Video(BaseModel):
    id: str
    duration: str
    categories: Optional[List[str]] = Field(
        default=None, description="List of categories associated with the video"
    )
    scenes: List[Scene]
    title: Optional[str] = Field(default=None, description="Title of the video")
    src: Optional[str] = Field(default=None, description="Source URL of the video")
    explanation: Optional[str] = Field(
        default=None, description="Explanation of the video"
    )
    code: Optional[str] = Field(
        default=None, description="Code associated with the video"
    )
    diagram: Optional[str] = Field(
        default=None, description="Diagram associated with the video"
    )
    approaches: Optional[List[Approach]] = Field(
        default=None, description="List of approaches associated with the video"
    )
    exercises: Optional[List[Exercise]] = Field(
        default=None, description="List of exercises associated with the video"
    )

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

        # This must be the desired amount + 1, as we start counting from 1
        images_per_scene = 4

        id: str = str(uuid4())
        duration = str(object=video.duration)
        scenes_list: list = []

        for i, (start_time, end_time) in enumerate(iterable=scenes, start=1):
            scene = Scene(
                start_timestamp=str(object=start_time.get_timecode()),
                end_timestamp=str(object=end_time.get_timecode()),
                images=[
                    ImageModel(
                        path=f"media/images/{id}/scenes/{id}-Scene-{i:03d}-{j:02d}.png",
                    )
                    for j in range(1, images_per_scene)
                ],
            )
            scenes_list.append(scene)

        return cls(id=id, duration=duration, scenes=scenes_list)
