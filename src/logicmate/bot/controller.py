import json
import logging
import os

from inference_sdk import InferenceHTTPClient

from logicmate.models.ia.pyscenedetect.psycenedetect import PySceneDetect
from logicmate.models.ia.surya.surya import Surya
from logicmate.models.ia.yolo.code_detector.code_detector import CodeDetector
from logicmate.models.ia.yolo.code_diagram_detector.code_diagram_detector import (
    CodeDiagramDetector,
)
from logicmate.models.video.video import Video
from logicmate.utils.directory.directory import DirectoryUtil
from logicmate.utils.file.file import FileUtil


def remove_similar_images(
    video: Video,
) -> Video:
    """
    Removes similar images from the video based on a similarity threshold.

    Args:
        video (Video): The video object to be processed.
        threshold (float): The similarity threshold for removing images.

    Returns:
        Video: The processed video with similar images removed.
    """
    # Placeholder for actual implementation


def predict_diagram_classification(
    video: Video,
    client: InferenceHTTPClient,
) -> Video:
    """
    Predicts the diagram classification for a given video.

    Args:
        video (Video): The video object to be processed.
        client (InferenceHTTPClient): The inference HTTP client for making predictions.

    Returns:
        Video: The processed video with predictions.
    """
    pass


def predict_drawio_diagram(
    video: Video,
    client: InferenceHTTPClient,
) -> Video:
    """
    Predicts the drawio diagram classification for a given video.

    Args:
        video (Video): The video object to be processed.
        client (InferenceHTTPClient): The inference HTTP client for making predictions.

    Returns:
        Video: The processed video with predictions.
    """
    pass


def predict_flowgorithm_diagram(
    video: Video,
    client: InferenceHTTPClient,
) -> Video:
    """
    Predicts the flowgorithm diagram classification for a given video.

    Args:
        video (Video): The video object to be processed.
        client (InferenceHTTPClient): The inference HTTP client for making predictions.

    Returns:
        Video: The processed video with predictions.
    """
    pass


def predict_code_snippet(
    video: Video,
    client: InferenceHTTPClient,
) -> Video:
    """
    Predicts the code snippet classification for a given video.

    Args:
        video (Video): The video object to be processed.
        client (InferenceHTTPClient): The inference HTTP client for making predictions.

    Returns:
        Video: The processed video with predictions.
    """
    code_detector: CodeDetector = CodeDetector(client=client)
    video = code_detector.predict_from_video(
        video=video,
        use_client=False,
        show_result=False,
    )
    return video


def extract_video_text(
    video: Video,
) -> Video:
    """
    Extracts text from the video.

    Args:
        video (Video): The video object to be processed.

    Returns:
        Video: The processed video with extracted text.
    """
    surya = Surya()
    video = surya.predict_video(video=video)

    return video


def predict_by_video_categories(
    video: Video,
    client: InferenceHTTPClient,
) -> Video:
    match video.categories:
        case ["code"]:
            video = predict_code_snippet(video=video, client=client)
        case ["diagram"]:
            video = predict_diagram_classification(video=video)
        case _:
            raise ValueError(f"Unknown category: {video.categories}")
    return video


def start_bot(path_to_file: str) -> None:
    if not path_to_file:
        raise ValueError("Path to file is required.")

    API_KEY: str | None = os.getenv(key="API_KEY")
    if not API_KEY:
        raise ValueError("API_KEY environment variable is not set.")
    API_URL: str | None = os.getenv(key="API_URL")
    if not API_URL:
        raise ValueError("API_URL environment variable is not set.")
    CLIENT = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)
    if not CLIENT:
        raise ValueError("Inference client is not initialized.")

    logging.info(msg="Starting the bot...")

    scene_detector: PySceneDetect = PySceneDetect()
    codeDiagramDetector: CodeDiagramDetector = CodeDiagramDetector(client=CLIENT)

    file_exist, file_path, file_name = FileUtil.file_exists(path=path_to_file)
    if not file_exist:
        raise FileNotFoundError(f"File not found: {path_to_file}")

    output_dir: str = DirectoryUtil.ensure_directory(
        path=f"media/images/{file_name}/scenes"
    )

    video: Video = scene_detector.process_video(
        video_path=file_path, output_dir=output_dir
    )

    if not video:
        raise ValueError("Video is None. Please check the video processing.")

    video = codeDiagramDetector.predict_from_video(
        video=video,
        use_client=False,
    )

    if not video:
        raise ValueError("Video is None. Please check the video processing.")
    if not video.categories:
        raise ValueError(
            "Video categories are None. Please check the video processing."
        )
    if not video.scenes:
        raise ValueError("Video scenes are None. Please check the video processing.")

    video = predict_by_video_categories(
        video=video,
        client=CLIENT,
    )

    if not video:
        raise ValueError("Video is None. Please check the video processing.")

    video = extract_video_text(video=video)
    logging.info(msg="Bot finished processing.")
