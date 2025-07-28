import json
import logging
import os
import shutil
from typing import FrozenSet, List, Literal, Set, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scenedetect import AdaptiveDetector, StatsManager, open_video, save_images
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.scene_detector import SceneDetector
from scenedetect.scene_manager import Interpolation, SceneManager
from scenedetect.video_stream import VideoStream

from logicmate.models.video.video import Video
from logicmate.utils.directory.directory import DirectoryUtil
from logicmate.utils.file.file import FileUtil

ValidImageExtensions = FrozenSet[Literal["jpg", "jpeg", "png", "webp"]]


class PySceneDetect(BaseModel):
    video: VideoStream = Field(default=None)
    scene_manager: SceneManager = Field(
        default_factory=lambda: SceneManager(stats_manager=StatsManager())
    )
    interpolation: Interpolation = Field(default_factory=lambda: Interpolation(value=4))
    valid_images_extensions: Set[str] = Field(
        default_factory=lambda: frozenset({"jpg", "jpeg", "png", "webp"})
    )
    detector: SceneDetector = Field(
        default_factory=lambda: AdaptiveDetector(
            adaptive_threshold=10.0,
            min_scene_len=10,
            window_width=6,
            min_content_val=10,
            luma_only=False,
            kernel_size=None,
        )
        # default_factory=lambda: AdaptiveDetector(
        #     adaptive_threshold=15.0,
        #     min_scene_len=20,
        #     window_width=8,
        #     min_content_val=12,
        #     luma_only=False,
        #     kernel_size=None,
        # )
        # default_factory=lambda: AdaptiveDetector(
        #     adaptive_threshold=10,
        #     min_scene_len=10,
        #     window_width=6,
        #     min_content_val=10,
        #     luma_only=True,
        #     kernel_size=None,
        # )
    )
    model_config: ConfigDict = {
        "arbitrary_types_allowed": True,
    }

    @model_validator(mode="after")
    def add_detector_to_scene_manager(self) -> "PySceneDetect":
        if self.scene_manager and self.detector:
            self.scene_manager.add_detector(detector=self.detector)
        return self

    def on_new_scene(self, _: np.ndarray, frame_num: int) -> None:
        """
        Callback function that is called when a new scene is detected.

        Args:
            _ (np.ndarray): The image frame where the scene was detected.
            frame_num (int): The frame number where the scene was detected.
        """
        logging.info(msg=f"Detected Scene: {frame_num} - {frame_num + 1}")

    def set_video(self, video: VideoStream) -> None:
        """
        Sets the video stream to be processed.

        Args:
            video (VideoStream): The video stream to be processed.
        """
        self.video = video
        logging.info(msg=f"Video set: {video}")

    def open_and_set_video(self, video_path: str) -> VideoStream:
        """
        Opens the video file, saves a copy to media/videos, and returns a VideoStream.

        Args:
            video_path (str): The path to the video file.

        Returns:
            VideoStream: The opened video stream.
        """
        video: VideoStream = open_video(path=video_path)
        self.set_video(video=video)

        videos_dir = DirectoryUtil.ensure_directory("media/videos")
        filename = os.path.basename(video_path)
        destination_path = os.path.join(videos_dir, filename)

        if not os.path.exists(destination_path):
            shutil.copy2(video_path, destination_path)
            logging.info(f"Video copied to: {destination_path}")
        else:
            logging.info(f"Video already exists at: {destination_path}")

        logging.info(f"Video opened: {video}")
        return video

    def detect_scenes(self) -> None:
        """
        Detects scenes in the video using the scene manager and detector.
        """
        if not self.video:
            raise ValueError(
                "Video not set. Please set the video before detecting scenes. Use `open_and_set_video()` method."
            )

        logging.info(msg="Detecting scenes...")
        self.scene_manager.detect_scenes(
            video=self.video, show_progress=True, callback=self.on_new_scene
        )
        logging.info(msg="Scenes detected.")

    def save_scenes(self, output_dir: str) -> None:
        """
        Saves the detected scenes as images in the specified output directory.

        Args:
            output_dir (str): The directory where the images will be saved.
        """
        if not self.video:
            raise ValueError(
                "Video not set. Please set the video before saving scenes. Use `open_and_set_video()` method."
            )

        logging.info(msg="Saving scenes...")
        scene_list: List[Tuple[FrameTimecode, FrameTimecode]] = (
            self.scene_manager.get_scene_list()
        )

        save_images(
            video=self.video,
            show_progress=True,
            output_dir=output_dir,
            interpolation=self.interpolation,
            scene_list=scene_list,
            image_extension="png",
            encoder_param=9,  # best for png
        )
        logging.info(msg=f"Saved {len(scene_list)} scenes.")

    def detect_and_save_scenes(self, output_dir: str) -> Video:
        """
        Detects scenes in the video and saves them as images in the specified output directory.

        Args:
            output_dir (str): The directory where the images will be saved.
        """
        if not self.video:
            raise ValueError(
                "Video not set. Please set the video before detecting scenes. Use `open_and_set_video()` method."
            )

        logging.info(msg="Detecting and saving scenes...")
        self.detect_scenes()
        self.save_scenes(output_dir=output_dir)
        logging.info(msg="Scenes detected and saved.")

        video_result: Video = Video.create_from_video_scenes(
            video=self.video,
            scenes=self.scene_manager.get_scene_list(),
        )

        return video_result

    def process_video(self, video_path: str, output_dir: str) -> Video:
        """
        Processes the video by detecting and saving scenes.

        Args:
            video_path (str): The path to the video file.
            output_dir (str): The directory where the images will be saved.

        Returns:
            Video: The processed video with detected scenes.
        """
        self.open_and_set_video(video_path=video_path)
        return self.detect_and_save_scenes(output_dir=output_dir)


if __name__ == "__main__":
    path_to_file = "media/videos/6c26a5d1-5f67-4752-b1a4-7b5911cdd157.mp4"

    # Example usage
    file_exist, file_path, file_name = FileUtil.file_exists(path=path_to_file)
    if not file_exist:
        raise FileNotFoundError(f"File not found: {path_to_file}")

    output_dir: str = DirectoryUtil.ensure_directory(
        path=f"media/images/{file_name}/scenes"
    )

    # Create a PySceneDetect instance
    scene_detector: PySceneDetect = PySceneDetect()

    # Process the video
    video: Video = scene_detector.process_video(
        video_path=file_path, output_dir=output_dir
    )

    print(json.dumps(video.model_dump(mode="json"), indent=4, ensure_ascii=False))
