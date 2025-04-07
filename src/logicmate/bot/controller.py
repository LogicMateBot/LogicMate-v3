import logging

from inference_sdk import InferenceHTTPClient

from logicmate.models.ia.dino.dino import Dino
from logicmate.models.ia.openai.openai import OpenAIModel
from logicmate.models.ia.phi.phi import Phi
from logicmate.models.ia.pyscenedetect.psycenedetect import PySceneDetect
from logicmate.models.ia.surya.surya import Surya
from logicmate.models.ia.yolo.code_detector.code_detector import CodeDetector
from logicmate.models.ia.yolo.code_diagram_detector.code_diagram_detector import (
    CodeDiagramDetector,
)
from logicmate.models.ia.yolo.diagram_type_detector.diagram_type_detector import (
    DiagramTypeDetector,
)
from logicmate.models.ia.yolo.drawio_detector.drawio_detector import DrawioDetector
from logicmate.models.video.video import Video
from logicmate.utils.directory.directory import DirectoryUtil
from logicmate.utils.env.env import EnvVariable
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
    dino: Dino = Dino(model_name="facebook/dinov2-base")
    video = dino.filter_images(video=video, threshold=0.995)
    return video


def predict_diagram_classification(
    video: Video,
    client: InferenceHTTPClient,
    use_client: bool = False,
    show_result: bool = False,
) -> Video:
    """
    Predicts the diagram classification for a given video.

    Args:
        video (Video): The video object to be processed.
        client (InferenceHTTPClient): The inference HTTP client for making predictions.

    Returns:
        Video: The processed video with predictions.
    """
    diagramTypeDetector: DiagramTypeDetector = DiagramTypeDetector(client=client)
    video = diagramTypeDetector.predict_from_video(
        video=video, use_client=use_client, show_result=show_result
    )

    if video.categories and "drawio" in video.categories:
        video = predict_drawio_diagram(
            video=video, client=client, use_client=use_client, show_result=show_result
        )
    if video.categories and "flowgorithm" in video.categories:
        video = predict_flowgorithm_diagram(
            video=video, client=client, use_client=use_client, show_result=show_result
        )

    return video


def predict_drawio_diagram(
    video: Video,
    client: InferenceHTTPClient,
    use_client: bool = False,
    show_result: bool = False,
) -> Video:
    """
    Predicts the drawio diagram classification for a given video.

    Args:
        video (Video): The video object to be processed.
        client (InferenceHTTPClient): The inference HTTP client for making predictions.

    Returns:
        Video: The processed video with predictions.
    """
    drawioDetector: DrawioDetector = DrawioDetector(client=client)
    video = drawioDetector.predict_from_video(
        video=video, use_client=use_client, show_result=show_result
    )
    return video


def predict_flowgorithm_diagram(
    video: Video,
    client: InferenceHTTPClient,
    use_client: bool = False,
    show_result: bool = False,
) -> Video:
    """
    Predicts the flowgorithm diagram classification for a given video.

    Args:
        video (Video): The video object to be processed.
        client (InferenceHTTPClient): The inference HTTP client for making predictions.

    Returns:
        Video: The processed video with predictions.
    """
    flowgorithmDetector: CodeDiagramDetector = CodeDiagramDetector(client=client)
    video = flowgorithmDetector.predict_from_video(
        video=video, use_client=use_client, show_result=show_result
    )
    return video


def predict_code_snippet(
    video: Video,
    client: InferenceHTTPClient,
    use_client: bool = False,
    show_result: bool = False,
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
        video=video, use_client=use_client, show_result=show_result
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
    use_client: bool = False,
    show_result: bool = False,
) -> Video:
    match video.categories:
        case ["code"]:
            video = predict_code_snippet(
                video=video,
                client=client,
                use_client=use_client,
                show_result=show_result,
            )
        case ["diagram"]:
            video = predict_diagram_classification(
                video=video,
                client=client,
                use_client=use_client,
                show_result=show_result,
            )
        case _:
            raise ValueError(f"Unknown category: {video.categories}")
    return video


def explain_video(
    video: Video,
    model_to_use: str,
    openai_api_key: str | None = None,
) -> Video:
    """
    Explains the video using the Phi model.

    Args:
        video (Video): The video object to be processed.


    Returns:
        Video: The processed video with explanations.
    """

    match model_to_use:
        case "phi":
            phi: Phi = Phi(model_name="microsoft/Phi-4-mini-instruct")
            video = phi.generate_explanation(video=video)
        case "openai":
            if not openai_api_key:
                raise ValueError("OpenAI API key is not set.")

            openai: OpenAIModel = OpenAIModel(api_key=openai_api_key)
            video = openai.generate_explanation(video=video)
        case _:
            raise ValueError(f"Unknown model: {model_to_use}")
    return video


def start_bot(video_path: str, config: dict[str, str]) -> None:
    if not video_path:
        raise ValueError("Path to file is required.")

    roboflow_api_key: str = config.get(EnvVariable.ROBOFLOW_API_KEY.value)
    roboflow_api_url: str = config.get(EnvVariable.ROBOFLOW_API_URL.value)
    openai_api_key: str = config.get(EnvVariable.OPENAI_API_KEY.value)

    CLIENT = InferenceHTTPClient(api_url=roboflow_api_url, api_key=roboflow_api_key)
    if not CLIENT:
        raise ValueError("Inference client is not initialized.")

    logging.info(msg="Starting the bot...")

    scene_detector: PySceneDetect = PySceneDetect()
    codeDiagramDetector: CodeDiagramDetector = CodeDiagramDetector(client=CLIENT)

    file_exist, file_path, file_name = FileUtil.file_exists(path=video_path)
    if not file_exist:
        raise FileNotFoundError(f"File not found: {video_path}")

    output_dir: str = DirectoryUtil.ensure_directory(
        path=f"media/images/{file_name}/scenes"
    )

    video: Video = scene_detector.process_video(
        video_path=file_path, output_dir=output_dir
    )

    video = codeDiagramDetector.predict_from_video(
        video=video,
        use_client=False,
        show_result=False,
    )

    if not video.categories:
        raise ValueError(
            "Video categories are None. Please check the video processing."
        )
    if not video.scenes:
        raise ValueError("Video scenes are None. Please check the video processing.")

    video = remove_similar_images(video=video)

    if not video:
        raise ValueError("Video is None. Please check the video processing.")

    video = predict_by_video_categories(
        video=video,
        client=CLIENT,
        use_client=False,
        show_result=False,
    )
    video = extract_video_text(video=video)

    video = explain_video(
        video=video, model_to_use="phi", openai_api_key=openai_api_key
    )

    logging.info(msg="Bot finished processing.")
