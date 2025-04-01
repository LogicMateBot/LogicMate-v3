from abc import ABC, abstractmethod
from typing import Any, List, Optional

from pydantic import BaseModel, Field, model_validator
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
        default=0.25,
        description="Confidence threshold for predictions.",
    )
    valid_classes: list[str] = Field(default=None, description="List of valid classes.")

    model_config = {
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
        formatted_predictions: list[PredictionBase] = []

        for i, (xyxy, class_name) in enumerate(
            zip(detections.xyxy, detections.data["class_name"])
        ):
            x_min, y_min, x_max, y_max = xyxy

            width = x_max - x_min
            height = y_max - y_min
            x_center = x_min + width / 2
            y_center = y_min + height / 2

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

    def sort_predictions_by_x(self, predictions: list[Any]) -> list[Any]:
        return sorted(predictions, key=lambda obj: obj["x"])

    def sort_predictions_by_y(self, predictions: list[Any]) -> list[Any]:
        return sorted(predictions, key=lambda obj: obj["y"])
