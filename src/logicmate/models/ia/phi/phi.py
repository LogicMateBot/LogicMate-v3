from logicmate.models.predictions.predictions.prediction import PredictionBase
from logicmate.models.video.video import ImageModel, Scene, Video
from pydantic import BaseModel, ConfigDict, Field, model_validator
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.pipelines.base import Pipeline


class Phi(BaseModel):
    """
    Class for the Phi model, a large language model (LLM) based on the LLaMA architecture.
    """

    model_name: str = Field(default=None, description="The name of the model.")
    model: AutoModelForCausalLM = Field(default=None, description="The model instance.")
    tokenizer: AutoTokenizer = Field(
        default=None, description="The tokenizer instance."
    )
    pipe: Pipeline = Field(default=None, description="The pipeline instance.")
    generation_args: dict = Field(
        default={
            "max_new_tokens": 2048,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        },
        description="Generation arguments for the model.",
    )

    model_config: ConfigDict = {
        "arbitrary_types_allowed": True,
    }

    @model_validator(mode="after")
    def load_model(self) -> "Phi":
        """
        Load the Phi model with the specified weights after initialization.
        """
        if self.model_name:
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.model_name
            )
            self.pipe: Pipeline = pipeline(
                task="text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
            )
        return self

    def generate_prediction_explanation(self, prediction: PredictionBase) -> str:
        """
        Generate an explanation for a prediction using the Phi model.
        Args:
            prediction (PredictionBase): The prediction object containing the prediction data.
        Returns:
            str: The generated explanation.
        """

        messages: list = [
            {
                "role": "developer",
                "content": (
                    "If the input is a diagram node, explain its meaning and how it fits into the diagram. "
                    "If it's a C language code snippet, explain its purpose and operation step-by-step. "
                    "Clearly highlight any errors or bad practices present. "
                    "Provide only the explanation without extra comments, formatting, or additional text."
                    "Response should be concise and focused on the explanation. Do not 'guess' the code. "
                    "Do not include the class name or the text in the answer. "
                    "Finally, answer must be in Spanish."
                ),
            },
            {
                "role": "user",
                "content": f"This is the class: {prediction.class_name} \n. This is the text: {prediction.text}",
            },
        ]

        pipe: Pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        explanation: str = pipe(inputs=messages, **self.generation_args)[0][
            "generated_text"
        ]

        return explanation

    def generate_image_explanation(self, image: ImageModel) -> str:
        """
        Generate an image explanation using the Phi model.

        Args:
            predictions (List[PredictionBase]): List of predictions associated with the image.

        Returns:
            str: The generated explanation.
        """
        combined_content: str = "\n".join(
            [
                f"This is the class: {prediction.class_name}\n. This is the text: {prediction.text}\n"
                for prediction in image.predictions
            ]
        )
        messages: list = [
            {
                "role": "developer",
                "content": (
                    "Given the following set of fragments, which may include code written in C language or flow diagrams, "
                    "clearly identify the general purpose of the image. Precisely explain what process, operation, or logic "
                    "is being performed or represented by these contents. If C code is identified, briefly explain its functionality, "
                    "highlighting any errors or poor practices. If flow diagrams are identified, briefly describe the represented process. "
                    "The response must be concise, accurate, without additional comments, and entirely in Spanish."
                ),
            },
            {
                "role": "user",
                "content": f"Extracted content from the image:\n{combined_content}",
            },
        ]

        pipe: Pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        explanation: str = pipe(inputs=messages, **self.generation_args)[0][
            "generated_text"
        ]

        return explanation

    def generate_scene_explanation(self, scene: Scene) -> str:
        """
        Generate a scene explanation using the Phi model.

        Args:
            images (List[ImageModel]): List of images associated with the scene.

        Returns:
            str: The generated explanation.
        """
        combined_explanations: str = "\n".join(
            [
                f"Image {index + 1} explanation: {image.explanation}"
                for index, image in enumerate(scene.images)
            ]
        )

        messages: list = [
            {
                "role": "developer",
                "content": (
                    "Based on the following set of explanations extracted from multiple images, clearly identify the overall purpose of the scene. "
                    "Precisely explain what process, operation, or logic the entire scene is representing or illustrating. "
                    "Highlight key relationships or connections between the individual image explanations. "
                    "The response must be concise, accurate, without additional comments, and entirely in Spanish."
                ),
            },
            {
                "role": "user",
                "content": f"Extracted explanations from the images in the scene:\n{combined_explanations}",
            },
        ]

        pipe: Pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        explanation: str = pipe(inputs=messages, **self.generation_args)[0][
            "generated_text"
        ]

        return explanation

    def generate_video_explanation(self, video: Video) -> str:
        """
        Generate a video explanation using the Phi model.

        Args:
            video (Video): The video object containing the video data.

        Returns:
            str: The generated explanation.
        """
        combined_explanations: str = "\n".join(
            [
                f"Scene {index + 1} explanation: {scene.explanation}"
                for index, scene in enumerate(iterable=video.scenes)
            ]
        )

        messages: list = [
            {
                "role": "developer",
                "content": (
                    "Based on the following set of explanations extracted from multiple scenes, clearly identify the overall purpose of the video. "
                    "Precisely explain what process, operation, or logic the entire video is representing or illustrating. "
                    "Highlight key relationships or connections between the individual scene explanations. ",
                    "Set a title for the video, it should be short and concise and represent the content of the video. ",
                    "The response must be concise, accurate, without additional comments, and entirely in Spanish.",
                ),
            },
            {
                "role": "user",
                "content": f"Extracted explanations from the scenes in the video:\n{combined_explanations}",
            },
        ]

        pipe: Pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        explanation: str = pipe(inputs=messages, **self.generation_args)[0][
            "generated_text"
        ]

        return explanation

    def generate_explanation(self, video: Video) -> Video:
        """
        Generate a video explanation using the Phi model.

        Args:
            video (Video): The video object containing the video data.

        Returns:
            str: The generated explanation.
        """

        for scene in video.scenes:
            for image in scene.images:
                for prediction in image.predictions:
                    prediction.explanation = self.generate_prediction_explanation(
                        prediction=prediction
                    )
                image.explanation = self.generate_image_explanation(image=image)
            scene.explanation = self.generate_scene_explanation(images=scene.images)
        video.explanation = self.generate_video_explanation(video=video)

        return video
