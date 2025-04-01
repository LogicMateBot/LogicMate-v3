from typing import Literal

from logicmate.models.predictions.predictions.prediction import PredictionBase


class DrawIOInitialNode(PredictionBase):
    prediction_class: Literal["initial-node"]
    text: Literal["Inicio"]


class DrawIOFinalNode(PredictionBase):
    prediction_class: Literal["final-node"]
    text: Literal["Fin"]


class DrawIOPrintNode(PredictionBase):
    prediction_class: Literal["print-node"]


class DrawIOVariableNode(PredictionBase):
    prediction_class: Literal["variable-node"]


class DrawIOOperationNode(PredictionBase):
    prediction_class: Literal["operation-node"]


class DrawIODecisionNode(PredictionBase):
    prediction_class: Literal["desicion-node"]


class DrawIOFunctionNode(PredictionBase):
    prediction_class: Literal["function-node"]


class DrawIODecisionArrow(PredictionBase):
    prediction_class: Literal["desicion-arrow"]


class DrawIONormalArrow(PredictionBase):
    prediction_class: Literal["normal-arrow"]
