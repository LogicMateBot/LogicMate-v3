import logging
from typing import List, Set

from pydantic import Field
from supervision.detection.core import Detections

from logicmate.models.ia.yolo.yolo.yolo import Yolo
from logicmate.models.predictions.predictions.prediction import PredictionBase
from logicmate.models.video.video import Video


class CodeDiagramDetector(Yolo):
    """
    Class for detecting code diagrams using YOLO model.
    Inherits from the Yolo class.
    """

    model_weight: str = "weights/code-diagram-detector-weights.pt"
    model_id: str = "code-vs-diagram/1"
    valid_classes: Set[str] = Field(
        default_factory=lambda: frozenset({"code", "diagram"})
    )

    def determine_image_category(self, predictions: list[PredictionBase]) -> str:
        code_count: int = sum(
            (1 for prediction in predictions if prediction.class_name == "code")
        )
        diagram_count: int = sum(
            (1 for prediction in predictions if prediction.class_name == "diagram")
        )

        return (
            "code"
            if code_count > diagram_count
            else "diagram"
            if diagram_count > code_count
            else "code_diagram"
        )

    def determine_scene_category(self, image_categories: List[str]) -> List[str]:
        code_votes: int = image_categories.count("code")
        diagram_votes: int = image_categories.count("diagram")

        if code_votes == 0 and diagram_votes == 0:
            return ["none"]

        if code_votes > diagram_votes and code_votes > 1:
            return ["code"]
        elif diagram_votes > code_votes and diagram_votes > 1:
            return ["diagram"]
        else:
            return ["none"]

    def determine_video_category(
        self, scene_categories: List[List[str]], total_scenes: int
    ) -> List[str]:
        code_votes: int = sum(
            (1 for category in scene_categories if "code" in category)
        )
        diagram_votes: int = sum(
            (1 for category in scene_categories if "diagram" in category)
        )
        video_categories: list = []
        threshold: float = total_scenes * 0.3
        if code_votes > threshold:
            video_categories.append("code")
        if diagram_votes > threshold:
            video_categories.append("diagram")
        return video_categories

    def filter_none_scene_category(self, video: Video) -> Video:
        """
        Filter out scenes with "none" category from the video.
        """
        logging.info(msg=f"Number of scenes: {len(video.scenes)} before filtering")
        video.scenes = [
            scene for scene in video.scenes if "none" not in scene.categories
        ]
        logging.info(msg=f"Number of scenes: {len(video.scenes)} after filtering")
        return video

    def filter_video_scenes_by_video_category(self, video: Video) -> Video:
        """
        Filter out scenes that do not match the video category.
        """
        logging.info(msg=f"Number of scenes: {len(video.scenes)} before filtering")
        video.scenes = [
            scene
            for scene in video.scenes
            if any((category in scene.categories for category in video.categories))
        ]
        logging.info(msg=f"Number of scenes: {len(video.scenes)} after filtering")
        return video

    def filter_images_by_scene_category(self, video: Video) -> Video:
        """
        Filter out images that do not match the scene category.
        """
        for scene in video.scenes:
            scene.images = [
                image
                for image in scene.images
                if any(category in image.categories for category in scene.categories)
            ]
        return video

    def parse_detections_to_predictions(
        self,
        detections: list,
    ) -> List[PredictionBase]:
        formatted_predictions: list[PredictionBase] = []

        for xyxy, class_name in zip(detections.xyxy, detections.data["class_name"]):
            x_min, y_min, x_max, y_max = xyxy

            width: float = x_max - x_min
            height: float = y_max - y_min
            x_center: float = x_min + width / 2
            y_center: float = y_min + height / 2

            prediction: PredictionBase = PredictionBase(
                x=x_center,
                y=y_center,
                width=width,
                height=height,
                text=None,
                explanation=None,
                class_name=class_name,
            )
            formatted_predictions.append(prediction)

        return formatted_predictions

    def parse_raw_predictions_from_client_to_prediction_objects(
        self,
        raw_predictions: list,
    ) -> List[PredictionBase]:
        """
        Converts raw predictions to prediction objects.
        Args:
            raw_predictions (list): List of raw predictions from the YOLO model.
        Returns:
            list: List of prediction objects.
        """
        formatted_predictions: list[PredictionBase] = []

        for prediction in raw_predictions:
            if prediction["confidence"] < self.confidence:
                continue

            prediction: PredictionBase = PredictionBase(
                x=prediction["x"],
                y=prediction["y"],
                width=prediction["width"],
                height=prediction["height"],
                text=None,
                explanation=None,
                class_name=prediction["class"],
            )
            formatted_predictions.append(prediction)

        return formatted_predictions

    def predict(
        self, image_path: str, show_result: bool = False
    ) -> List[PredictionBase]:
        """
        Predicts the categories of images in the
        video using the YOLO model.
        Args:
            image_path (str): Path to the image to predict.
            show_result (bool): Whether to show the result.
        Returns:
            List[PredictionBase]: A list of predictions for the image.
        """

        ultralytics_results = self.model(image_path, conf=self.confidence)[0]
        detections_results: Detections = Detections.from_ultralytics(
            ultralytics_results=ultralytics_results
        ).with_nms()
        predictions: List[PredictionBase] = self.parse_detections_to_predictions(
            detections=detections_results
        )

        if show_result:
            self.show_image_with_predictions(
                image_path=image_path,
                detections=detections_results,
            )

        return predictions

    def predict_with_client(
        self, image_path: str, show_result: bool = False
    ) -> List[PredictionBase]:
        """
        Predicts the categories of images in the
        video using the YOLO model.
        Args:
            image_path (str): Path to the image to predict.
            show_result (bool): Whether to show the result.
        Returns:
            List[PredictionBase]: A list of predictions for the image.
        """
        if not self.client:
            logging.error(msg="Inference client is not initialized.")
            raise ValueError("Inference client is not initialized.")
        if not self.model_id:
            logging.error(msg="Model ID is not set.")
            raise ValueError("Model ID is not set.")
        if not image_path:
            logging.error(msg="Image path is not provided.")
            raise ValueError("Image path is not provided.")

        logging.info(msg=f"Predicting with client: {self.client}, image: {image_path}")
        results: dict | List[dict] = self.client.infer(
            inference_input=image_path,
            model_id=self.model_id,
        )

        raw_predictions: dict = results.get("predictions", [])
        predictions: List[PredictionBase] = (
            self.parse_raw_predictions_from_client_to_prediction_objects(
                raw_predictions=raw_predictions
            )
        )

        if show_result:
            self.show_image_with_predictions_client(
                image_path=image_path,
                predictions=raw_predictions,
            )
        logging.info(msg=f"Image: {image_path}, predicted.")
        return predictions

    def predict_from_video(
        self, video: Video, use_client: bool = False, show_result: bool = False
    ) -> Video:
        # This should remain outside because later we will determine the main video category
        scene_categories: list = []

        for scene in video.scenes:
            image_categories: list = []
            for image in scene.images:
                if use_client:
                    image.predictions = self.predict_with_client(
                        image_path=image.path, show_result=show_result
                    )
                else:
                    image.predictions = self.predict(
                        image_path=image.path, show_result=show_result
                    )
                if len(image.predictions) == 0:
                    image.categories = ["none"]
                    image_categories.append("none")
                    continue

                category: str = self.determine_image_category(
                    predictions=image.predictions
                )
                if category in ["code", "diagram"]:
                    image.categories = [category]
                    image_categories.append(category)
                else:
                    image.categories = ["code", "diagram"]
                    image_categories.append("code")
                    image_categories.append("diagram")

            scene_category: List[str] = self.determine_scene_category(
                image_categories=image_categories
            )
            scene.categories = scene_category
            scene_categories.append(scene_category)

        video.categories = self.determine_video_category(
            scene_categories=scene_categories, total_scenes=len(video.scenes)
        )

        video = self.filter_none_scene_category(video=video)
        video = self.filter_video_scenes_by_video_category(video=video)
        video = self.filter_images_by_scene_category(video=video)
        return video
