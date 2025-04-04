import logging
from typing import List

from pydantic import Field
from supervision.detection.core import Detections

from logicmate.models.ia.yolo.yolo.yolo import Yolo
from logicmate.models.predictions.code_predictions.code_predictions import (
    CodeBracket,
    CodePrediction,
    CodeSnippet,
)
from logicmate.models.video.video import Video


class CodeDetector(Yolo):
    """
    Class for detecting code snippets using YOLO model.
    Inherits from the Yolo class.
    """

    model_weight: str = "weights-v1/code-snippet-detection-model/weights/best.pt"
    model_id: str = "code-snippet-video-class-detection/27"
    min_amount_of_predictions: int = 4
    valid_classes: set[str] = Field(
        default_factory=lambda: frozenset({"code_snippet", "code-bracket"})
    )

    def filter_images_from_scene_by_min_amount_of_predictions(
        self, video: Video
    ) -> Video:
        """
        Filters images from the scene based on the minimum amount of predictions.

        Args:
            video (Video): The video object to be processed.

        Returns:
            Video: The processed video with filtered images.
        """
        for scene in video.scenes:
            scene.images = [
                image
                for image in scene.images
                if len(image.predictions) >= self.min_amount_of_predictions
            ]
        return video

    def parse_detections_to_predictions(
        self,
        detections: list,
    ) -> List[CodePrediction]:
        """
        Abstract method to format detections into predictions.

        Args:
            detections (list): List of detections from the YOLO model.

        Returns:
            list: Formatted predictions.
        """
        formatted_predictions: list[CodePrediction] = []

        for xyxy, class_name in zip(detections.xyxy, detections.data["class_name"]):
            x_min, y_min, x_max, y_max = xyxy

            width: float = x_max - x_min
            height: float = y_max - y_min
            x_center: float = x_min + width / 2
            y_center: float = y_min + height / 2

            match class_name:
                case "code_snippet":
                    prediction = CodeSnippet(
                        x=x_center,
                        y=y_center,
                        width=width,
                        height=height,
                    )
                case "code_bracket":
                    prediction = CodeBracket(
                        x=x_center,
                        y=y_center,
                        width=width,
                        height=height,
                    )
                case _:
                    pass

            formatted_predictions.append(prediction)

        return formatted_predictions

    def parse_raw_predictions_to_prediction_objects(
        self,
        raw_predictions: list,
    ) -> List[CodePrediction]:
        """
        Converts raw predictions to prediction objects.
        Args:
            raw_predictions (list): List of raw predictions from the YOLO model.
        Returns:
            list: List of prediction objects.
        """
        formatted_predictions: list[CodePrediction] = []

        for prediction in raw_predictions:
            if prediction["confidence"] < self.confidence:
                continue

            match prediction["class"]:
                case "code_snippet":
                    prediction = CodeSnippet(
                        x=prediction["x"],
                        y=prediction["y"],
                        width=prediction["width"],
                        height=prediction["height"],
                    )
                case "code_bracket":
                    prediction = CodeBracket(
                        x=prediction["x"],
                        y=prediction["y"],
                        width=prediction["width"],
                        height=prediction["height"],
                    )
                case _:
                    pass

            formatted_predictions.append(prediction)

        return formatted_predictions

    def predict(
        self, image_path: str, show_result: bool = False
    ) -> List[CodePrediction]:
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
        detections_results = Detections.from_ultralytics(
            ultralytics_results=ultralytics_results
        ).with_nms()
        predictions: List[CodeSnippet | CodeBracket] = (
            self.parse_detections_to_predictions(detections=detections_results)
        )

        if show_result:
            self.show_image_with_predictions(
                image_path=image_path,
                detections=detections_results,
            )

        return predictions

    def predict_with_client(
        self, image_path, show_result=False
    ) -> list | List[CodePrediction]:
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
        predictions: List[CodePrediction] = (
            self.parse_raw_predictions_to_prediction_objects(
                raw_predictions=raw_predictions
            )
        )

        if predictions is None:
            logging.warning(
                msg=f"Image: {image_path} has no predictions or not enough predictions."
            )
            return []
        if len(predictions) < self.min_amount_of_predictions:
            logging.warning(
                msg=f"Image: {image_path} has no predictions or not enough predictions."
            )
            return []

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
        """
        Predicts code snippets from a video using the YOLO model.

        Args:
            video (Video): The video object to be processed.

        Returns:
            Video: The processed video with predictions.
        """
        for scene in video.scenes:
            for image in scene.images:
                if use_client:
                    image.predictions = self.predict_with_client(
                        image_path=image.path, show_result=show_result
                    )
                else:
                    image.predictions = self.predict(
                        image_path=image.path, show_result=show_result
                    )

                if (
                    image.predictions is None
                    or len(image.predictions) < self.min_amount_of_predictions
                ):
                    logging.warning(
                        msg=f"Image: {image.path} has no predictions or not enough predictions."
                    )
                    continue

                image.predictions = self.sort_predictions_by_y(
                    predictions=image.predictions
                )

        video = self.filter_images_from_scene_by_min_amount_of_predictions(
            video=video,
        )

        return video
