from typing import Optional
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod


class PredictionBase(BaseModel, ABC):
    x: float
    y: float
    text: Optional[str] = Field(
        default=None, description="Text associated with the prediction"
    )
    explanation: Optional[str] = Field(
        default=None, description="Explanation of the prediction"
    )
    prediction_class: str

    @abstractmethod
    def process(self):
        """Process the prediction data."""
        pass
