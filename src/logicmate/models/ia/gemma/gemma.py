import logging
from io import BytesIO
import re
from typing import List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from PIL import Image
import ollama
from pydantic import BaseModel, ConfigDict, Field

from logicmate.models.video.video import Video
from logicmate.models.video.video import ImageModel, Scene
from logicmate.models.predictions.predictions.prediction import PredictionBase


class TextLine(BaseModel):
    """A single line of recognized text with metadata."""

    text: str
    bbox: List[float]
    confidence: float


class OCRResult(BaseModel):
    """OCR result containing one or more recognized text lines."""

    text_lines: List[TextLine]


class Gemma3OCR(BaseModel):
    model: str = Field(
        default="gemma3:12b", description="Name of the Ollama model to use for OCR"
    )

    prompt: str = Field(
        default="give only the text that you see in the image, no additional commentary",
        description="Instruction for the Ollama model",
    )

    # Classes to ignore when processing videos
    ignore_classes: set[str] = Field(
        default_factory=lambda: frozenset(
            {"initial-node", "final-node", "normal-arrow", "code_bracket"}
        ),
        description="Classes to ignore in video predictions",
    )

    model_config: ConfigDict = {"arbitrary_types_allowed": True}

    def image_to_bytes(
        self, image: Union[bytes, Image.Image, np.ndarray], format: str = "PNG"
    ) -> bytes:
        if isinstance(image, bytes):
            return image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(image, Image.Image):
            buf = BytesIO()
            image.save(fp=buf, format=format)
            return buf.getvalue()
        raise TypeError(f"Unsupported image type: {type(image)}")

    def get_predictions(
        self, image: Union[bytes, ndarray, Image.Image]
    ) -> List[OCRResult]:
        img_bytes = self.image_to_bytes(image)
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": self.prompt,
                        "images": [img_bytes],
                    }
                ],
            )
            text = response.message.content or ""
        except Exception as e:
            raise RuntimeError(f"Error calling ollama.chat: {e}")

        line = TextLine(text=text, bbox=[0.0, 0.0, 0.0, 0.0], confidence=1.0)
        return [OCRResult(text_lines=[line])]

    def get_text_from_predictions(self, predictions: List[OCRResult]) -> str:
        if not predictions:
            raise ValueError("Predictions are required")
        first = predictions[0].text_lines
        if not first:
            return ""
        return first[0].text

    def get_bounding_boxes_from_predictions(
        self, predictions: List[OCRResult]
    ) -> List[float]:
        if not predictions:
            raise ValueError("Predictions are required")
        first = predictions[0].text_lines
        if not first:
            return [0.0, 0.0, 0.0, 0.0]
        return first[0].bbox

    def get_confidence_from_predictions(self, predictions: List[OCRResult]) -> float:
        if not predictions:
            raise ValueError("Predictions are required")
        first = predictions[0].text_lines
        if not first:
            return 0.0
        return first[0].confidence

    def show_image_with_bounding_boxes(
        self, image: ndarray, bounding_boxes: List[float], text: str = "Detected Text"
    ) -> None:
        if not bounding_boxes:
            raise ValueError("Bounding boxes are required")
        x1, y1, x2, y2 = map(int, bounding_boxes)
        cv2.rectangle(
            image=image,
            pt1=(x1, y1),
            pt2=(x2, y2),
            color=(0, 255, 0),
            thickness=2,
        )
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(text)
        plt.show()

    def clean_class_name(self, raw: str) -> str:
        match = re.match(r"typing\.Literal\['(.+)'\]", raw)
        return match.group(1) if match else raw

    def ignore_classes_for_ocr_prediction(self, class_name: str) -> bool:
        return class_name in self.ignore_classes

    def predict(
        self, image: ndarray, show_result: bool = False
    ) -> Tuple[str, List[float], float]:
        predictions = self.get_predictions(image)
        text = self.get_text_from_predictions(predictions)
        bboxes = self.get_bounding_boxes_from_predictions(predictions)
        conf = self.get_confidence_from_predictions(predictions)
        if show_result:
            self.show_image_with_bounding_boxes(image, bboxes, text)
            logging.info(
                msg=f"Predicted text: {text}, Bounding boxes: {bboxes}, Confidence: {conf}"
            )
        return text, bboxes, conf

    def cropped_image_from_predictions(
        self,
        image_path: str,
        predictions: PredictionBase,
        show_image: bool = False,
        show_text: bool = False,
    ) -> ndarray:
        if not image_path:
            raise ValueError("Image path is required")
        if not predictions:
            raise ValueError("Predictions are required")
        image = cv2.imread(image_path)
        x_center, y_center = predictions.x, predictions.y
        w, h = predictions.width, predictions.height
        x1 = max(0, int(x_center - w / 2))
        y1 = max(0, int(y_center - h / 2))
        x2 = min(image.shape[1], int(x_center + w / 2))
        y2 = min(image.shape[0], int(y_center + h / 2))
        cropped = image[y1:y2, x1:x2]
        # Upscale & convert to BGR three channels
        scaled = cv2.resize(cropped, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        ch_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if show_text or show_image:
            text = None
            if show_text:
                text, _, _ = self.predict(ch_image)
            plt.imshow(cv2.cvtColor(ch_image, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title(text if show_text else "Cropped Image")
            plt.show()
        return cropped

    def predict_video(
        self, video: Video, show_result: bool, show_image: bool, show_text: bool
    ) -> Video:
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

                    not_should_validate = self.ignore_classes_for_ocr_prediction(
                        class_name=self.clean_class_name(raw=str(prediction.class_name))
                    )
                    if not_should_validate:
                        logging.info(
                            msg=f"This class name: {prediction.class_name}. Already have text."
                        )
                        continue

                    logging.info(msg=f"Processing prediction: {prediction}")
                    cropped = self.cropped_image_from_predictions(
                        image_path=image.path,
                        predictions=prediction,
                        show_image=show_image,
                        show_text=show_text,
                    )
                    text, _, _ = self.predict(
                        image=cropped,
                        show_result=show_result,
                    )
                    prediction.text = text
        return video


class Gemma3(BaseModel):
    """
    Client for interacting with an Ollama model to generate explanations and reconstruct C code
    from OCR/predictions.
    """

    model_for_explications: str = Field(
        default="gemma3:12b",
        description="Ollama model to use for generating explanations.",
    )
    model_for_recreations: str = Field(
        default="gemma3:12b", description="Ollama model to use for reconstructing code."
    )

    def generate_prediction_explanation(self, prediction: PredictionBase) -> str:
        """
        Generate a concise Spanish explanation for a single prediction fragment.

        Uses a specialized prompt per class_name to tailor the explanation.
        """
        base_instruction = (
            "Provide an explanation in Spanish for the given content. "
            "Do not include the class name or the original text in your response. "
            "The answer should be concise, direct, and without extra formatting or comments.\n\n"
        )
        class_instructions = {
            "code_snippet": (
                "The input is a single line of C code. Explain clearly and step by step what it does "
                "and how it works. Do not mention errors, bad practices, or suggest improvements."
            ),
            "code_bracket": (
                "The input is an opening or closing bracket ('{' or '}') in a C program. "
                "Explain its purpose in structuring code blocks."
            ),
            "initial-node": (
                "This is a start node in a flowchart. Explain what it represents and how it begins the process."
            ),
            "final-node": (
                "This is an end node in a flowchart. Explain its purpose in marking the end of the process."
            ),
            "print-node": (
                "This is a print/output node in a flowchart. Explain its role in displaying information during the process."
            ),
            "variable-node": (
                "This node represents a variable declaration or initialization. Explain its role in the algorithm."
            ),
            "operation-node": (
                "This node performs an operation. Describe what kind of operation it might be and how it affects the process."
            ),
            "decision-node": (
                "This node represents a conditional decision (like an 'if'). Explain how it influences the program flow."
            ),
            "function-node": (
                "This node represents a function call or definition. Explain its purpose within the process."
            ),
            "input-node": (
                "This node represents user input in a flowchart. Describe its importance in the logic of the algorithm."
            ),
            "output-node": (
                "This node represents data output. Explain how it is used to show results or messages to the user."
            ),
            "normal-arrow": (
                "This is a regular arrow in a flowchart. Explain how it connects steps in the process."
            ),
            "decision-arrow": (
                "This is a decision arrow (Yes/No) in a flowchart. Explain how it directs the program flow based on conditions."
            ),
        }

        specific_instruction = class_instructions.get(prediction.class_name, "")
        full_prompt = base_instruction + specific_instruction
        text_content = prediction.text or ""

        response = ollama.chat(
            model=self.model_for_explications,
            messages=[
                {"role": "developer", "content": full_prompt},
                {"role": "user", "content": text_content},
            ],
        )
        explanation = (
            response.message.content.strip() if response.message.content else ""
        )
        prediction.explanation = explanation
        return explanation

    def generate_image_explanation(self, image: ImageModel) -> str:
        """
        Generate a concise Spanish summary of what an image represents,
        based on its code or diagram fragments and their explanations.
        """
        if not image.predictions:
            raise ValueError(
                "Image must contain predictions to generate an explanation."
            )

        categories = image.categories or []
        if not categories:
            raise ValueError(
                "Image must have at least one category (e.g., 'code' or 'diagram')."
            )

        combined_content = "\n".join(
            [
                f"Fragment {i + 1}:\nText: {p.text or '[no text]'}\n"
                f"Explanation: {p.explanation or '[no explanation]'}"
                for i, p in enumerate(image.predictions)
            ]
        )

        if "code" in categories:
            role_prompt = (
                "You are analyzing a set of C language code fragments extracted from an image. "
                "Based on the explanations and code lines, generate a concise description in Spanish "
                "explaining what the entire program or logic does. "
                "Use the explanations as support, but do not repeat them. "
                "Do not include extra commentary. Focus on summarizing what the code accomplishes as a whole."
            )
        elif "diagram" in categories:
            role_prompt = (
                "You are analyzing a set of flowchart elements extracted from an image. "
                "Based on the node texts and their explanations, generate a concise description in Spanish "
                "that summarizes the overall logic or process represented by the diagram. "
                "Use the explanations as support, but do not repeat them. "
                "Do not include extra commentary or formatting."
            )
        else:
            role_prompt = (
                "You are analyzing mixed content (code fragments and/or diagrams). "
                "Generate a concise explanation in Spanish about what is represented, "
                "focusing on the overall logic and purpose. Base your answer on the provided texts "
                "and explanations without repeating them directly."
            )

        response = ollama.chat(
            model=self.model_for_explications,
            messages=[
                {"role": "developer", "content": role_prompt},
                {
                    "role": "user",
                    "content": f"Extracted content from the image:\n{combined_content}",
                },
            ],
        )
        explanation = (
            response.message.content.strip() if response.message.content else ""
        )
        image.explanation = explanation
        return explanation

    def generate_scene_explanation(self, scene: Scene) -> str:
        """
        Generate a concise Spanish summary of what a scene represents,
        based on the explanations of its constituent images.
        """
        if not scene.images:
            raise ValueError("Scene must contain images.")

        if any(image.explanation is None for image in scene.images):
            raise ValueError("All images must have generated explanations first.")

        categories = scene.categories or []
        combined_explanations = "\n".join(
            [
                f"Image {i + 1}:\nExplanation: {image.explanation}"
                for i, image in enumerate(scene.images)
            ]
        )

        if "code" in categories and "diagram" in categories:
            role_prompt = (
                "You are analyzing a process represented through both C code fragments and flow diagrams. "
                "Summarize the overall logic and objective of the entire scene, "
                "explaining how the diagrams and code complement each other. "
                "Do not repeat image explanations. Respond in Spanish, concisely."
            )
        elif "code" in categories:
            role_prompt = (
                "You are analyzing a scene composed of C code fragments. "
                "Describe in Spanish what the full program or algorithm accomplishes, "
                "without repeating individual image explanations. Be concise."
            )
        elif "diagram" in categories:
            role_prompt = (
                "You are analyzing a scene composed of flowchart diagrams. "
                "Explain in Spanish the general process or algorithm being represented, "
                "without repeating individual explanations. Be concise."
            )
        else:
            role_prompt = (
                "You are analyzing mixed content. Summarize in Spanish the overall purpose "
                "or logic of the scene based on the extracted explanations. Be concise."
            )

        response = ollama.chat(
            model=self.model_for_explications,
            messages=[
                {"role": "developer", "content": role_prompt},
                {
                    "role": "user",
                    "content": f"Explanations extracted from the scene:\n{combined_explanations}",
                },
            ],
        )
        explanation = (
            response.message.content.strip() if response.message.content else ""
        )
        scene.explanation = explanation
        return explanation

    def generate_video_explanation(self, video: Video) -> str:
        """
        Generate a concise Spanish summary and a short Spanish title for the entire video,
        based on its scene explanations.
        """
        if not video.scenes:
            raise ValueError("Video must contain scenes.")

        if any(scene.explanation is None for scene in video.scenes):
            raise ValueError("All scenes must have explanations first.")

        combined = "\n".join(
            [
                f"Scene {i + 1}:\nExplanation: {scene.explanation}"
                for i, scene in enumerate(video.scenes)
            ]
        )
        categories = video.categories or []

        if "code" in categories and "diagram" in categories:
            role_prompt = (
                "You are analyzing a video composed of both C code and flow diagrams. "
                "Summarize in Spanish what the video as a whole represents, "
                "explaining how code and diagrams work together. "
                "Then provide a short Spanish title reflecting the content. Be concise."
            )
        elif "code" in categories:
            role_prompt = (
                "You are analyzing a video composed of C code scenes. "
                "Summarize in Spanish the complete program or algorithm presented, "
                "then give a short Spanish title. Be concise."
            )
        elif "diagram" in categories:
            role_prompt = (
                "You are analyzing a video composed of flowchart scenes. "
                "Summarize in Spanish the overall process illustrated, "
                "then give a short Spanish title. Be concise."
            )
        else:
            role_prompt = (
                "You are analyzing a video of mixed content. "
                "Summarize in Spanish the overall purpose or logic, "
                "then give a short Spanish title. Be concise."
            )

        response = ollama.chat(
            model=self.model_for_explications,
            messages=[
                {"role": "developer", "content": role_prompt},
                {
                    "role": "user",
                    "content": f"Scene explanations from the video:\n{combined}",
                },
            ],
        )
        explanation = (
            response.message.content.strip() if response.message.content else ""
        )
        video.explanation = explanation
        return explanation

    def generate_code_from_video(self, video: Video) -> str:
        """
        Reconstruct a full, compilable C program in proper order and formatting
        from fragmented code predictions within the video.
        """
        if not video.scenes:
            raise ValueError("Video must contain scenes.")

        fragments = []
        for scene in video.scenes:
            for image in scene.images:
                for pred in image.predictions:
                    if (
                        pred.class_name in ("code_snippet", "code_bracket")
                        and pred.text
                    ):
                        fragments.append(pred.text.strip())

        if not fragments:
            raise ValueError("No code-related fragments found.")

        raw_input = "\n".join(fragments)
        prompt = (
            "You are given a sequence of fragmented C code lines. "
            "Reconstruct the full C program in correct order and formatting, following this pattern:\n\n"
            "- Include necessary libraries (e.g., <stdio.h>, <stdlib.h>, <conio.h>)\n"
            "- Define macros if needed\n"
            "- Declare function signatures\n"
            "- Implement the main() function\n"
            "- Document each function using comments in English\n"
            "- Add the complete function definitions\n\n"
            "Use only the provided fragments. Preserve variable names, logic, and formatting. "
            "Output only the code."
        )

        response = ollama.chat(
            model=self.model_for_recreations,
            messages=[
                {"role": "developer", "content": prompt},
                {
                    "role": "user",
                    "content": f"Here are the extracted code fragments:\n{raw_input}",
                },
            ],
        )
        final_code = (
            response.message.content.strip() if response.message.content else ""
        )
        video.code = final_code
        return final_code

    def generate_explanation(self, video: Video) -> Video:
        """
        Run the full pipeline: explanations for predictions, images, scenes, video,
        and reconstruct code if missing.
        """
        for scene in video.scenes:
            for image in scene.images:
                for pred in image.predictions:
                    pred.explanation = self.generate_prediction_explanation(pred)
                image.explanation = self.generate_image_explanation(image)
            scene.explanation = self.generate_scene_explanation(scene)
        video.explanation = self.generate_video_explanation(video)
        if video.code is None:
            video.code = self.generate_code_from_video(video)
        return video
