from typing import Literal, Union

from pydantic import Field

from logicmate.models.predictions.predictions.prediction import PredictionBase


class CodeSnippet(PredictionBase):
    class_name: Literal["code_snippet"] = Field(default="code_snippet")


class CodeBracket(PredictionBase):
    class_name: Literal["code_bracket"] = Field(default="code_bracket")
    text: Literal["{ ó }"] = Field(
        default="{ ó }",
        description="Text associated with the code bracket prediction.",
    )


CodePrediction = Union[
    CodeSnippet,
    CodeBracket,
]
