from typing import Literal

from logicmate.models.predictions.predictions.prediction import PredictionBase


class FlowgorithmInitialNode(PredictionBase):
    prediction_class: Literal["initial-node"]
    text: Literal["Inicio"]


class FlowgorithmFinalNode(PredictionBase):
    prediction_class: Literal["final-node"]
    text: Literal["Fin"]


class FlowgorithmInputNode(PredictionBase):
    prediction_class: Literal["input-node"]


class FlowgorithmOutputNode(PredictionBase):
    prediction_class: Literal["output-node"]


class FlowgorithmOperationNode(PredictionBase):
    prediction_class: Literal["operation-node"]


class FlowgorithmDecisionNode(PredictionBase):
    prediction_class: Literal["decision-node"]


class FlowgorithmNormalArrow(PredictionBase):
    prediction_class: Literal["normal-arrow"]


class FlowgorithmDecisionArrow(PredictionBase):
    prediction_class: Literal["decision-arrow"]
