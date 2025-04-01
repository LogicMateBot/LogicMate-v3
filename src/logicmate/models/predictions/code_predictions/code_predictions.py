from typing import Literal, Union

from logicmate.models.predictions.predictions.prediction import PredictionBase


class CodeSnippet(PredictionBase):
    prediction_class: Literal["code_snippet"]


class CodeBracket(PredictionBase):
    prediction_class: Literal["code-bracket"]
    text: Literal["{ o }"]


CodePrediction = Union[
    CodeSnippet,
    CodeBracket,
]
