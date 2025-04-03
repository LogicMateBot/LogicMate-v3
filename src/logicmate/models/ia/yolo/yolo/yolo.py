from abc import ABC, abstractmethod
from typing import List, Optional

import cv2
from numpy import ndarray
import supervision as sv
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ultralytics import YOLO

from logicmate.models.predictions.predictions.prediction import PredictionBase
from logicmate.models.video.video import Video


class Yolo(BaseModel, ABC):
    """
    Base class for YOLO (You Only Look Once) models.
    This class serves as a blueprint for implementing specific YOLO models.
    """

    model: Optional[YOLO] = Field(
        default=None,
        description="YOLO model instance. This will be initialized with the model_weight.",
    )
    model_weight: str = Field(
        default=None,
        description="Path to the YOLO model weights.",
    )
    confidence: float = Field(
        default=0.50,
        description="Confidence threshold for predictions.",
    )
    client: Optional[InferenceHTTPClient] = Field(
        default=None,
        description="Inference HTTP client for making predictions.",
    )
    valid_classes: list[str] = Field(default=None, description="List of valid classes.")

    model_config: ConfigDict = {
        "arbitrary_types_allowed": True,
    }

    @model_validator(mode="after")
    def load_model(self) -> "Yolo":
        """
        Load the YOLO model with the specified weights after initialization.
        """
        if self.model_weight:
            self.model = YOLO(model=self.model_weight)
        return self

    @abstractmethod
    def parse_detections_to_predictions(
        self,
        detections: list,
    ) -> List[PredictionBase]:
        """
        Abstract method to format detections into predictions.

        Args:
            detections (list): List of detections from the YOLO model.

        Returns:
            list: Formatted predictions.
        """
        pass

    @abstractmethod
    def parse_raw_predictions_to_prediction_objects(
        self,
        raw_predictions: list,
    ) -> List[PredictionBase]:
        """
        Abstract method to convert raw predictions to prediction objects.

        Args:
            raw_predictions (list): List of raw predictions from the YOLO model.

        Returns:
            list: List of prediction objects.
        """
        pass

    @abstractmethod
    def predict_from_video(
        self,
        video: Video,
    ) -> Video:
        """
        Abstract method to predict from a video.

        Args:
            video (Video): Video object to process.

        Returns:
            Video: Processed video with predictions.
        """
        pass

    @abstractmethod
    def predict(
        self, image_path: str, show_result: bool = False
    ) -> List[PredictionBase]:
        """
        Predicts the categories of images in the
        video using the YOLO model.
        Args:
            video (Video): The video object containing the images to predict.
        Returns:
            List[PredictionBase]: A list of predictions for each image in the video.
        """
        pass

    @abstractmethod
    def predict_with_client(
        self, image_path: str, show_result: bool = False
    ) -> List[PredictionBase]:
        """
        Predicts the categories of images in the
        video using the YOLO model.
        Args:
            video (Video): The video object containing the images to predict.
        Returns:
            List[PredictionBase]: A list of predictions for each image in the video.
        """
        pass

    def sort_predictions_by_x(
        self, predictions: list[PredictionBase]
    ) -> list[PredictionBase]:
        return sorted(iterable=predictions, key=lambda obj: obj.x)

    def sort_predictions_by_y(
        self, predictions: list[PredictionBase]
    ) -> list[PredictionBase]:
        return sorted(iterable=predictions, key=lambda obj: obj.y)

    def show_image_with_predictions(
        self,
        image_path: str,
        detections,
    ) -> None:
        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        image = cv2.imread(filename=image_path)
        annotated_image: ndarray = bounding_box_annotator.annotate(
            scene=image, detections=detections
        )
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections
        )

        sv.plot_image(image=annotated_image)

    def show_image_with_predictions_client(
        self,
        image_path: str,
        predictions: dict,
    ) -> None:
        image = Image.open(fp=image_path)
        draw = ImageDraw.Draw(im=image)

        for bounding_box in predictions:
            x = bounding_box["x"]
            y = bounding_box["y"]
            width = bounding_box["width"]
            height = bounding_box["height"]

            x1 = min(x - width / 2, x + width / 2)
            x2 = max(x - width / 2, x + width / 2)
            y1 = min(y - height / 2, y + height / 2)
            y2 = max(y - height / 2, y + height / 2)
            box = (x1, y1, x2, y2)

            draw.rectangle(xy=box, outline="red", width=2)

        image.show()
