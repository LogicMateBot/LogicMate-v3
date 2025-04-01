import json
import logging

from logicmate.models.ia.pyscenedetect.psycenedetect import PySceneDetect
from logicmate.models.ia.yolo.code_diagram_detector.code_diagram_detector import (
    CodeDiagramDetector,
)
from logicmate.models.video.video import Video
from logicmate.utils.directory.directory import DirectoryUtil
from logicmate.utils.file.file import FileUtil


def start_bot(path_to_file: str) -> None:
    """
    Starts the bot by initializing the necessary components and running the main loop.
    """
    logging.info("Starting the bot...")

    scene_detector: PySceneDetect = PySceneDetect()
    codeDiagramDetector: CodeDiagramDetector = CodeDiagramDetector()

    file_exist, file_path, file_name = FileUtil.file_exists(path=path_to_file)
    if not file_exist:
        raise FileNotFoundError(f"File not found: {path_to_file}")

    output_dir: str = DirectoryUtil.ensure_directory(f"media/images/{file_name}/scenes")

    video: Video = scene_detector.process_video(
        video_path=file_path, output_dir=output_dir
    )

    print(json.dumps(video.model_dump(mode="json"), indent=4, ensure_ascii=False))

    video = codeDiagramDetector.predict_from_video(
        video=video,
    )

    print(json.dumps(video.model_dump(mode="json"), indent=4, ensure_ascii=False))
