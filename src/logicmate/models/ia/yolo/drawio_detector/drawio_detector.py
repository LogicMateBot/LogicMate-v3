import logging
from typing import List

from pydantic import Field
from supervision.detection.core import Detections

from logicmate.models.ia.yolo.yolo.yolo import Yolo
from logicmate.models.predictions.drawio_predictions.drawio_predictions import (
    DrawIODecisionArrow,
    DrawIODecisionNode,
    DrawIOFinalNode,
    DrawIOFunctionNode,
    DrawIOInitialNode,
    DrawIONormalArrow,
    DrawIOOperationNode,
    DrawioPrediction,
    DrawIOPrintNode,
    DrawIOVariableNode,
)
from logicmate.models.video.video import Video


class DrawioDetector(Yolo):
    """
    Class for detecting drawio diagrams using YOLO model.
    Inherits from the Yolo class.
    """

    model_weight: str = "weights-v1/drawio-diagram-detection-model/weights/best.pt"
    model_id: str = "drawio-videoclass-detection/6"
    min_amount_of_predictions: int = 2
    valid_classes: set[str] = Field(
        default_factory=lambda: frozenset(
            {
                "inital-node",
                "initial-node",
                "variable-node",
                "variable-declaration",
                "operation-node",
                "decision-node",
                "desicion-node",
                "function-node",
                "print-node",
                "final-node",
                "decision-arrow",
                "desicion-arrow",
                "normal-arrow",
            }
        )
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
    ) -> List[DrawioPrediction]:
        """
        Abstract method to format detections into predictions.

        Args:
            detections (list): List of detections from the YOLO model.

        Returns:
            list: Formatted predictions.
        """
        formatted_predictions: list[DrawioPrediction] = []

        for xyxy, class_name in zip(detections.xyxy, detections.data["class_name"]):
            x_min, y_min, x_max, y_max = xyxy

            width: float = x_max - x_min
            height: float = y_max - y_min
            x_center: float = x_min + width / 2
            y_center: float = y_min + height / 2

            match class_name:
                case "initial-node":
                    prediction: DrawIOInitialNode = DrawIOInitialNode(
                        x=x_center,
                        y=y_center,
                        width=width,
                        height=height,
                    )
                case "inital-node":
                    prediction: DrawIOInitialNode = DrawIOInitialNode(
                        x=x_center,
                        y=y_center,
                        width=width,
                        height=height,
                    )
                case "variable-declaration":
                    prediction: DrawIOVariableNode = DrawIOVariableNode(
                        x=x_center,
                        y=y_center,
                        width=width,
                        height=height,
                    )
                case "variable-node":
                    prediction: DrawIOVariableNode = DrawIOVariableNode(
                        x=x_center,
                        y=y_center,
                        width=width,
                        height=height,
                    )
                case "operation-node":
                    prediction: DrawIOOperationNode = DrawIOOperationNode(
                        x=x_center,
                        y=y_center,
                        width=width,
                        height=height,
                        class_name="operation-node",
                    )
                case "decision-node":
                    prediction: DrawIODecisionNode = DrawIODecisionNode(
                        x=x_center,
                        y=y_center,
                        width=width,
                        height=height,
                    )
                case "desicion-node":
                    prediction: DrawIODecisionNode = DrawIODecisionNode(
                        x=x_center,
                        y=y_center,
                        width=width,
                        height=height,
                    )
                case "function-node":
                    prediction: DrawIOFunctionNode = DrawIOFunctionNode(
                        x=x_center,
                        y=y_center,
                        width=width,
                        height=height,
                    )
                case "print-node":
                    prediction: DrawIOPrintNode = DrawIOPrintNode(
                        x=x_center,
                        y=y_center,
                        width=width,
                        height=height,
                    )
                case "final-node":
                    prediction: DrawIOFinalNode = DrawIOFinalNode(
                        x=x_center,
                        y=y_center,
                        width=width,
                        height=height,
                    )
                case "decision-arrow":
                    prediction: DrawIODecisionArrow = DrawIODecisionArrow(
                        x=x_center,
                        y=y_center,
                        width=width,
                        height=height,
                    )
                case "desicion-arrow":
                    prediction: DrawIODecisionArrow = DrawIODecisionArrow(
                        x=x_center,
                        y=y_center,
                        width=width,
                        height=height,
                    )
                case "normal-arrow":
                    prediction: DrawIONormalArrow = DrawIONormalArrow(
                        x=x_center,
                        y=y_center,
                        width=width,
                        height=height,
                    )
                case _:
                    raise ValueError(f"Invalid class name: {class_name}")

            formatted_predictions.append(prediction)
        return formatted_predictions

    def parse_raw_predictions_from_client_to_prediction_objects(
        self,
        raw_predictions: list,
    ) -> List[DrawioPrediction]:
        """
        Converts raw predictions to prediction objects.

        Args:
            raw_predictions (list): List of raw predictions from the YOLO model.

        Returns:
            list: List of prediction objects.
        """
        formatted_predictions: list[DrawioPrediction] = []

        for prediction in raw_predictions:
            if prediction["confidence"] < self.confidence:
                continue

            match prediction["class"]:
                case "inital-node":
                    prediction = DrawIOInitialNode(
                        x=prediction["x"],
                        y=prediction["y"],
                        width=prediction["width"],
                        height=prediction["height"],
                    )
                case "initial-node":
                    prediction = DrawIOInitialNode(
                        x=prediction["x"],
                        y=prediction["y"],
                        width=prediction["width"],
                        height=prediction["height"],
                    )
                case "variable-node":
                    prediction = DrawIOVariableNode(
                        x=prediction["x"],
                        y=prediction["y"],
                        width=prediction["width"],
                        height=prediction["height"],
                    )
                case "variable-declaration":
                    prediction = DrawIOVariableNode(
                        x=prediction["x"],
                        y=prediction["y"],
                        width=prediction["width"],
                        height=prediction["height"],
                    )
                case "operation-node":
                    prediction = DrawIOOperationNode(
                        x=prediction["x"],
                        y=prediction["y"],
                        width=prediction["width"],
                        height=prediction["height"],
                    )
                case "desicion-node":
                    prediction = DrawIODecisionNode(
                        x=prediction["x"],
                        y=prediction["y"],
                        width=prediction["width"],
                        height=prediction["height"],
                    )
                case "decision-node":
                    prediction = DrawIODecisionNode(
                        x=prediction["x"],
                        y=prediction["y"],
                        width=prediction["width"],
                        height=prediction["height"],
                    )
                case "function-node":
                    prediction = DrawIOFunctionNode(
                        x=prediction["x"],
                        y=prediction["y"],
                        width=prediction["width"],
                        height=prediction["height"],
                    )
                case "print-node":
                    prediction = DrawIOPrintNode(
                        x=prediction["x"],
                        y=prediction["y"],
                        width=prediction["width"],
                        height=prediction["height"],
                    )
                case "final-node":
                    prediction = DrawIOFinalNode(
                        x=prediction["x"],
                        y=prediction["y"],
                        width=prediction["width"],
                        height=prediction["height"],
                    )
                case "desicion-arrow":
                    prediction = DrawIODecisionArrow(
                        x=prediction["x"],
                        y=prediction["y"],
                        width=prediction["width"],
                        height=prediction["height"],
                    )
                case "decision-arrow":
                    prediction = DrawIODecisionArrow(
                        x=prediction["x"],
                        y=prediction["y"],
                        width=prediction["width"],
                        height=prediction["height"],
                    )
                case "normal-arrow":
                    prediction = DrawIONormalArrow(
                        x=prediction["x"],
                        y=prediction["y"],
                        width=prediction["width"],
                        height=prediction["height"],
                    )
                case _:
                    raise ValueError(f"Invalid class name: {prediction['class']}")

            formatted_predictions.append(prediction)
        return formatted_predictions

    def filter_empty_scenes(self, video: Video) -> Video:
        """
        Filter empty scenes from the video.
        Args:
            video (Video): Video object to process.
        Returns:
            Video: Video object with empty scenes removed.
        """
        if not video:
            raise ValueError("Video is required")
        if not video.scenes:
            raise ValueError("Scenes are required")

        logging.info(msg="Filtering empty scenes...")
        logging.info(msg=f"Number of scenes: {len(video.scenes)} before filtering")
        filtered_scenes: list = [
            scene for scene in video.scenes if scene.images and len(scene.images) > 0
        ]

        video.scenes = filtered_scenes
        logging.info(msg=f"Number of scenes: {len(video.scenes)} after filtering")
        logging.info(msg="Empty scenes filtered.")
        return video

    def predict(
        self, image_path: str, show_result: bool = False
    ) -> List[DrawioPrediction]:
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
        predictions: List[DrawioPrediction] = self.parse_detections_to_predictions(
            detections=detections_results
        )

        if show_result:
            self.show_image_with_predictions(
                image_path=image_path,
                detections=detections_results,
            )

        return predictions

    def predict_with_client(
        self, image_path, show_result=False
    ) -> list | List[DrawioPrediction]:
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
        predictions: List[DrawioPrediction] = (
            self.parse_raw_predictions_from_client_to_prediction_objects(
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
        logging.info(msg="Predicting drawio diagrams...")
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

        video = self.filter_empty_scenes(video=video)
        logging.info(msg="Drawio diagrams predicted.")

        return video
