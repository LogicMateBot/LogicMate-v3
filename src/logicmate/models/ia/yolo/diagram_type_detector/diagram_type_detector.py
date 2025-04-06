import logging
from typing import List

from pydantic import Field
from supervision.detection.core import Detections

from logicmate.models.ia.yolo.yolo.yolo import Yolo
from logicmate.models.predictions.predictions.prediction import PredictionBase
from logicmate.models.video.video import Video


class DiagramTypeDetector(Yolo):
    """
    Class for detecting diagram types using YOLO model.
    Inherits from the Yolo class.
    """

    model_weight: str = "weights-v1/diagram-classification-model/weights/best.pt"
    model_id: str = "drawio-vs-flowgorithm/5"
    min_amount_of_predictions: int = 1
    valid_classes: set[str] = Field(
        default_factory=lambda: frozenset({"drawio", "flowgorithm"})
    )

    def filter_none_scene_category(self, video: Video) -> Video:
        """
        Filter out scenes with "none" category from the video.
        """
        video.scenes = [
            scene for scene in video.scenes if "none" not in scene.categories
        ]
        return video

    def filter_video_scenes_by_video_category(self, video: Video) -> Video:
        """
        Filter out scenes that do not match the video category.
        """
        video.scenes = [
            scene
            for scene in video.scenes
            if any((category in scene.categories for category in video.categories))
        ]
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

    def determine_image_category(self, predictions: list[PredictionBase]) -> str:
        drawio_count: int = sum(
            (1 for prediction in predictions if prediction.class_name == "drawio")
        )
        flowgorithm_count: int = sum(
            (1 for prediction in predictions if prediction.class_name == "flowgorithm")
        )

        return (
            "drawio"
            if drawio_count > flowgorithm_count
            else "flowgorithm"
            if flowgorithm_count > drawio_count
            else "none"
        )

    def determine_scene_category(self, image_categories: List[str]) -> List[str]:
        drawio_votes: int = image_categories.count("drawio")
        flowgorithm_votes: int = image_categories.count("flowgorithm")

        if drawio_votes == 0 and flowgorithm_votes == 0:
            return ["none"]

        if drawio_votes > flowgorithm_votes and drawio_votes > 1:
            return ["drawio"]
        elif flowgorithm_votes > drawio_votes and flowgorithm_votes > 1:
            return ["flowgorithm"]
        else:
            return ["none"]

    def determine_video_category(
        self, scene_categories: List[List[str]], total_scenes: int
    ) -> List[str]:
        code_votes: int = sum(
            (1 for category in scene_categories if "drawio" in category)
        )
        diagram_votes: int = sum(
            (1 for category in scene_categories if "flowgorithm" in category)
        )
        video_categories: list = []
        threshold: float = total_scenes * 0.3
        if code_votes > threshold:
            video_categories.append("drawio")
        if diagram_votes > threshold:
            video_categories.append("flowgorithm")
        return video_categories

    def parse_detections_to_predictions(self, detections: list) -> List[PredictionBase]:
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
                class_name=class_name,
            )

            formatted_predictions.append(prediction)

        return formatted_predictions

    def parse_raw_predictions_from_client_to_prediction_objects(
        self, raw_predictions: list
    ) -> list:
        """
        Converts raw predictions to prediction objects.
        Args:
            raw_predictions (list): List of raw predictions from the YOLO model.
        Returns:
            list: List of prediction objects.
        """
        formatted_predictions: list[PredictionBase] = []

        for prediction in raw_predictions:
            x_min, y_min, x_max, y_max = prediction["bbox"]

            width: float = x_max - x_min
            height: float = y_max - y_min
            x_center: float = x_min + width / 2
            y_center: float = y_min + height / 2

            prediction_object: PredictionBase = PredictionBase(
                x=x_center,
                y=y_center,
                width=width,
                height=height,
                class_name=prediction["class_name"],
            )

            formatted_predictions.append(prediction_object)

        return formatted_predictions

    def predict(self, image_path, show_result=False):
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

    def predict_with_client(self, image_path, show_result=False):
        if not self.client:
            logging.error(msg="Inference client is not set.")
            raise ValueError("Inference client is not set.")
        if not self.model_id:
            logging.error(msg="Model ID is not set.")
            raise ValueError("Model ID is not set.")
        if not image_path:
            logging.error(msg="Image path is not set.")
            raise ValueError("Image path is not set.")

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
        if not video:
            logging.error(msg="Video object is not set.")
            raise ValueError("Video object is not set.")
        if not video.scenes:
            logging.error(msg="Video scenes are not set.")
            raise ValueError("Video scenes are not set.")

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
                    image.predictions = ["none"]
                    image_categories.append("none")
                    continue

                category: str = self.determine_image_category(
                    predictions=image.predictions
                )

                if category in ["drawio", "flowgorithm"]:
                    image.categories = [category]
                    image_categories.append(category)
                else:
                    image.categories = ["drawio", "flowgorithm"]
                    image_categories.append("drawio")
                    image_categories.append("flowgorithm")

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
