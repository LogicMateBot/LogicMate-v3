from typing import Literal, Union

from pydantic import Field

from logicmate.models.predictions.predictions.prediction import PredictionBase


class FlowgorithmInitialNode(PredictionBase):
    class_name: Literal["initial-node"] = Field(default=Literal["initial-node"])
    text: Literal["Inicio"] = Field(
        default="Inicio",
        description="Text associated with the initial node prediction.",
    )


class FlowgorithmFinalNode(PredictionBase):
    class_name: Literal["final-node"] = Field(default=Literal["final-node"])
    text: Literal["Fin"] = Field(
        default="Fin",
        description="Text associated with the final node prediction.",
    )


class FlowgorithmInputNode(PredictionBase):
    class_name: Literal["input-node"] = Field(default=Literal["input-node"])


class FlowgorithmOutputNode(PredictionBase):
    class_name: Literal["output-node"] = Field(default=Literal["output-node"])


class FlowgorithmOperationNode(PredictionBase):
    class_name: Literal["operation-node"] = Field(default=Literal["operation-node"])


class FlowgorithmDecisionNode(PredictionBase):
    class_name: Literal["decision-node"] = Field(default=Literal["decision-node"])


class FlowgorithmNormalArrow(PredictionBase):
    class_name: Literal["normal-arrow"] = Field(default=Literal["normal-arrow"])


class FlowgorithmDecisionArrow(PredictionBase):
    class_name: Literal["decision-arrow"] = Field(default=Literal["decision-arrow"])


FlowgorithmPrediction = Union[
    FlowgorithmInitialNode,
    FlowgorithmFinalNode,
    FlowgorithmInputNode,
    FlowgorithmOutputNode,
    FlowgorithmOperationNode,
    FlowgorithmDecisionNode,
    FlowgorithmNormalArrow,
    FlowgorithmDecisionArrow,
]
