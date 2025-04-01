from typing import List, Set

from pydantic import Field
from supervision.detection.core import Detections

from logicmate.models.ia.yolo.yolo.yolo import Yolo
from logicmate.models.video.video import Video
from logicmate.models.predictions.predictions.prediction import PredictionBase


class CodeDiagramDetector(Yolo):
    """
    Class for detecting code diagrams using YOLO model.
    Inherits from the Yolo class.
    """

    model_weight: str = "model/weights/code-diagram-detection-model/weights/best.pt"
    valid_classes: Set[str] = Field(
        default_factory=lambda: frozenset({"code", "diagram"})
    )

    def determine_image_category(self, predictions: list[PredictionBase]) -> str:
        code_count = sum(
            1 for prediction in predictions if prediction.class_name == "code"
        )
        diagram_count = sum(
            1 for prediction in predictions if prediction.class_name == "diagram"
        )

        return (
            "code"
            if code_count > diagram_count
            else "diagram"
            if diagram_count > code_count
            else "code_diagram"
        )

    def determine_scene_category(self, image_categories: List[str]) -> List[str]:
        code_votes = image_categories.count("code")
        diagram_votes = image_categories.count("diagram")

        if code_votes == 0 and diagram_votes == 0:
            return ["none"]

        if code_votes > diagram_votes:
            return ["code"]
        elif diagram_votes > code_votes:
            return ["diagram"]
        else:
            return ["code", "diagram"]

    def determine_video_category(
        self, scene_categories: List[List[str]], total_scenes: int
    ) -> List[str]:
        code_votes = sum(1 for category in scene_categories if "code" in category)
        diagram_votes = sum(1 for category in scene_categories if "diagram" in category)
        video_categories = []
        if code_votes > (total_scenes / 2):
            video_categories.append("code")
        if diagram_votes > (total_scenes / 2):
            video_categories.append("diagram")
        return video_categories

    def predict(self, image_path: str) -> List[PredictionBase]:
        """
        Predicts the categories of images in the
        video using the YOLO model.
        Args:
            video (Video): The video object containing the images to predict.
        Returns:
            List[PredictionBase]: A list of predictions for each image in the video.
        """
        ultralytics_results = self.model(image_path)[0]
        detections_results = Detections.from_ultralytics(
            ultralytics_results=ultralytics_results
        ).with_nms()
        predictions = self.parse_detections_to_predictions(
            detections=detections_results
        )

        return predictions

    def predict_from_video(self, video: Video) -> Video:
        scene_categories = []

        for scene in video.scenes:
            image_categories = []
            for image in scene.images:
                image.predictions = self.predict(image.path)
                if len(image.predictions) == 0:
                    image.categories = ["none"]
                    image_categories.append("none")
                    continue

                category = self.determine_image_category(image.predictions)
                if category in ["code", "diagram"]:
                    image_categories.append(category)
                else:
                    image_categories.append("code")
                    image_categories.append("diagram")

            scene_category = self.determine_scene_category(image_categories)
            scene.categories = scene_category
            scene_categories.append(scene_category)

        video.categories = self.determine_video_category(
            scene_categories, len(video.scenes)
        )
        return video
