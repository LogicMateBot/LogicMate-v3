from typing import Any

from numpy import ndarray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.metrics.pairwise import cosine_similarity
from torch import no_grad
from transformers import AutoImageProcessor, AutoModel

from logicmate.models.video.video import Video


class Dino(BaseModel):
    model_name: str
    processor: AutoImageProcessor = Field(
        default=None,
        description="Processor for the model.",
    )
    model: AutoModel = Field(
        default=None,
        description="Model for the model.",
    )
    model_config: ConfigDict = {
        "arbitrary_types_allowed": True,
    }

    @model_validator(mode="after")
    def load_model(self) -> "Dino":
        """
        Load the model with the specified name after initialization.
        """
        if self.model_name:
            self.processor = AutoImageProcessor.from_pretrained(
                pretrained_model_name_or_path=self.model_name
            )
            self.model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=self.model_name
            )
        return self

    def extract_features(self, image_path: str) -> list:
        image: Image.Image = Image.open(fp=image_path).convert(mode="RGB")
        inputs = self.processor(images=image, return_tensors="pt")

        with no_grad():
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()

        return features

    def get_similarity(self, features) -> ndarray:
        return cosine_similarity(X=features)

    def filter_similar_images(
        self, features, similarity_matrix, threshold=0.995
    ) -> list:
        amount_of_features: int = len(features)
        selected_indices: list = []

        for feature in range(amount_of_features):
            keep = True
            for index in selected_indices:
                if similarity_matrix[feature, index] > threshold:
                    keep = False
                    break
            if keep:
                selected_indices.append(feature)
        return selected_indices

    def get_images_features_from_video(self, video: Video) -> list:
        """
        Get image features from the video using the model.
        Args:
            video (Video): Video object to process.
        Returns:
            Video: Video object with image features.
        """
        if not video:
            raise ValueError("Video is required")
        if not video.scenes:
            raise ValueError("Scenes are required")

        feature_list: list = []

        for scene in video.scenes:
            for image in scene.images:
                image_features: list = self.extract_features(image_path=image.path)
                feature_list.append(image_features)

        flattened_feature_list: list = [feature.flatten() for feature in feature_list]
        return flattened_feature_list

    def filter_images(self, video: Video) -> Video:
        features: list = self.get_images_features_from_video(video=video)
        similarity_matrix: ndarray = self.get_similarity(features=features)
        selected_indices: list = self.filter_similar_images(
            features=features, similarity_matrix=similarity_matrix
        )

        images_to_remove: list = []

        index: int = 0
        for scene in video.scenes:
            for image in scene.images:
                if index not in selected_indices:
                    images_to_remove.append(image)
                image.features = features[index]
                index += 1

        for scene in video.scenes:
            scene.images = [
                image for image in scene.images if image not in images_to_remove
            ]

        return video
