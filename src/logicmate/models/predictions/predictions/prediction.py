from abc import ABC
from typing import Optional

from pydantic import BaseModel, Field


class PredictionBase(BaseModel, ABC):
    x: float
    y: float
    width: float
    height: float
    text: Optional[str] = Field(
        default=None, description="Text associated with the prediction"
    )
    explanation: Optional[str] = Field(
        default=None, description="Explanation of the prediction"
    )
    class_name: Optional[str] = Field(
        default=None,
        description="Class name of the prediction. This is used for classification tasks.",
    )
