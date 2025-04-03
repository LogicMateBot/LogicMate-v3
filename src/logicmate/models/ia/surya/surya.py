import logging
import os
from io import BytesIO
from typing import Any, List, Literal, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from surya.detection import DetectionPredictor
from surya.recognition import OCRResult, RecognitionPredictor

import logicmate.models.video.video as Video
from logicmate.models.predictions.predictions.prediction import PredictionBase


class Surya(BaseModel):
    os.environ["RECOGNITION_BATCH_SIZE"] = "512"
    detection_predictor: DetectionPredictor = Field(
        default_factory=DetectionPredictor,
        description="Detection predictor for text detection.",
    )
    recognition_predictor: RecognitionPredictor = Field(
        default_factory=RecognitionPredictor,
        description="Recognition predictor for text recognition.",
    )
    langs: Literal["es", "en"] = Field(
        default=["es", "en"],
        description="Language for text recognition. Supported languages: 'es', 'en'.",
    )
    model_config: ConfigDict = {
        "arbitrary_types_allowed": True,
    }

    def image_to_bytes(self, image: Image.Image, format: str = "PNG") -> bytes:
        buf = BytesIO()
        image.save(fp=buf, format=format)
        return buf.getvalue()

    def get_predictions(self, image) -> List[OCRResult]:
        if isinstance(obj=image, class_or_tuple=bytes):
            image: Image.Image = Image.open(fp=BytesIO(initial_bytes=image))

        elif isinstance(obj=image, class_or_tuple=np.ndarray):
            image = Image.fromarray(obj=image)

        elif not isinstance(obj=image, class_or_tuple=Image.Image):
            raise TypeError("Tipo de imagen no soportado")

        return self.recognition_predictor(
            images=[image], langs=[self.langs], det_predictor=self.detection_predictor
        )

    def get_text_from_predictions(self, predictions: List[OCRResult]) -> str:
        if not predictions:
            raise ValueError("Predictions are required")
        if not predictions[0].text_lines:
            return "No text detected."
        return predictions[0].text_lines[0].text

    def get_bounding_boxes_from_predictions(
        self, predictions: List[OCRResult]
    ) -> List[float]:
        if not predictions:
            raise ValueError("Predictions are required")
        if not predictions[0].text_lines:
            return [0, 0, 0, 0]
        return predictions[0].text_lines[0].bbox

    def get_confidence_from_predictions(self, predictions: List[OCRResult]) -> float:
        if not predictions:
            raise ValueError("Predictions are required")
        if not predictions[0].text_lines:
            return 0
        confidence: float | None = predictions[0].text_lines[0].confidence
        return confidence if confidence is not None else 0

    def show_image_with_bounding_boxes(
        self,
        image: ndarray,
        bounding_boxes: List[float],
        text: str = "Image with boundings",
    ) -> None:
        if not bounding_boxes:
            raise ValueError("Bounding boxes are required")

        x1, y1, x2, y2 = map(func=int, iterable=bounding_boxes)

        cv2.rectangle(
            img=image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2
        )

        plt.imshow(X=cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB))
        plt.axis(arg="off")
        plt.title(label=text)
        plt.show()

    def predict(
        self, image: ndarray, show_result: bool = False
    ) -> Tuple[str, List[float], float]:
        predictions: List[OCRResult] = self.get_predictions(image=image)
        extracted_text: str = self.get_text_from_predictions(predictions=predictions)
        bounding_boxes: List[float] = self.get_bounding_boxes_from_predictions(
            predictions=predictions
        )
        confidence: float = self.get_confidence_from_predictions(
            predictions=predictions
        )

        if show_result:
            self.show_image_with_bounding_boxes(
                image=image,
                bounding_boxes=bounding_boxes,
                text=extracted_text,
            )

        return extracted_text, bounding_boxes, confidence

    def predict_video(self, video: Video) -> Tuple[str, List[float], float]:
        if not video:
            raise ValueError("Video is required")
        if not video.scenes:
            raise ValueError("Scene images are required")

        for scene in video.scenes:
            logging.info(msg=f"Processing scene: {scene.scene_id}")
            for image in scene.images:
                if not image.path:
                    raise ValueError("Image path is required")
                logging.info(msg=f"Processing image: {image.path}")
                for prediction in image.predictions:
                    if not prediction:
                        raise ValueError("Prediction is required")

                    logging.info(msg=f"Processing prediction: {prediction}")
                    cropped_image_from_prediction: ndarray = (
                        self.cropped_image_from_predictions(
                            image_path=image.path,
                            predictions=prediction,
                            show_image=False,
                            show_text=False,
                        )
                    )
                    extracted_text, _, _ = self.predict(
                        image=cropped_image_from_prediction,
                        show_result=False,
                    )

                    prediction.text = extracted_text

        return video

    def cropped_image_from_predictions(
        self,
        image_path: str,
        predictions: PredictionBase,
        show_image: bool = False,
        show_text: bool = False,
    ) -> ndarray:
        if not image_path:
            raise ValueError("Image is required")
        if not predictions:
            raise ValueError("Predictions are required")

        image: ndarray = cv2.imread(filename=image_path)

        x_center: Any = predictions.x
        y_center: Any = predictions.y
        w: Any = predictions.width
        h: Any = predictions.height

        x1: int = int(x=x_center - w / 2)
        y1: int = int(x=y_center - h / 2)
        x2: int = int(x=x_center + w / 2)
        y2: int = int(x=y_center + h / 2)

        x1 = max(arg1=0, arg2=x1)
        y1 = max(arg1=0, arg2=y1)
        x2 = min(arg1=image.shape[1], arg2=x2)
        y2 = min(arg1=image.shape[0], arg2=y2)

        scale_factor = 2

        cropped_image: ndarray = image[y1:y2, x1:x2]
        rescaled_image = cv2.resize(
            src=cropped_image,
            dsize=None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_LANCZOS4,
        )
        gray_image = cv2.cvtColor(src=rescaled_image, code=cv2.COLOR_BGR2GRAY)
        channles_image = cv2.cvtColor(src=gray_image, code=cv2.COLOR_GRAY2BGR)

        if show_text:
            text, _, _ = self.predict(image=channles_image)

        if show_image:
            plt.imshow(X=channles_image)
            if show_text:
                plt.title(label=text)
            else:
                plt.title(label="Cropped Image")
            plt.axis(arg="off")
            plt.show()

        return cropped_image
