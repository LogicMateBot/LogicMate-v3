import logging
from typing import Literal, Union

from logicmate.models.predictions.predictions.prediction import PredictionBase


class CodeSnippet(PredictionBase):
    prediction_class: Literal["code_snippet"]

    def process(self):
        logging.info("Processing code snippet prediction")


class CodeBracket(PredictionBase):
    prediction_class: Literal["code-bracket"]
    text: Literal["{ o }"]

    def process(self):
        logging.info("Processing code bracket prediction")


CodePrediction = Union[
    CodeSnippet,
    CodeBracket,
]
