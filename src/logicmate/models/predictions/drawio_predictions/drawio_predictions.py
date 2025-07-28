from typing import Literal, Union

from pydantic import Field

from logicmate.models.predictions.predictions.prediction import PredictionBase


class DrawIOInitialNode(PredictionBase):
    class_name: Literal["initial-node"] = Field(default="initial-node")
    text: Literal["Inicio"] = Field(
        default="Inicio",
        description="Text associated with the initial node prediction.",
    )


class DrawIOFinalNode(PredictionBase):
    class_name: Literal["final-node"] = Field(default="final-node")
    text: Literal["Fin"] = Field(
        default="Fin",
        description="Text associated with the final node prediction.",
    )


class DrawIOPrintNode(PredictionBase):
    class_name: Literal["print-node"] = Field(default="print-node")


class DrawIOVariableNode(PredictionBase):
    class_name: Literal["variable-node"] = Field(default="variable-node")


class DrawIOOperationNode(PredictionBase):
    class_name: Literal["operation-node"] = Field(default="operation-node")


class DrawIODecisionNode(PredictionBase):
    class_name: Literal["decision-node"] = Field(default="decision-node")


class DrawIOFunctionNode(PredictionBase):
    class_name: Literal["function-node"] = Field(default="function-node")


class DrawIODecisionArrow(PredictionBase):
    class_name: Literal["decision-arrow"] = Field(default="decision-arrow")


class DrawIONormalArrow(PredictionBase):
    class_name: Literal["normal-arrow"] = Field(default="normal-arrow")


DrawioPrediction = Union[
    DrawIOInitialNode,
    DrawIOFinalNode,
    DrawIOPrintNode,
    DrawIOVariableNode,
    DrawIOOperationNode,
    DrawIODecisionNode,
    DrawIOFunctionNode,
    DrawIODecisionArrow,
    DrawIONormalArrow,
]
