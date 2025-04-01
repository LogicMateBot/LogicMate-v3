import logging

from typing import Literal
from logicmate.models.predictions.predictions.prediction import PredictionBase


class DrawIOInitialNode(PredictionBase):
    prediction_class: Literal["initial-node"]
    text: Literal["Inicio"]

    def process(self):
        logging.info("Processing drawio initial node prediction")


class DrawIOFinalNode(PredictionBase):
    prediction_class: Literal["final-node"]
    text: Literal["Fin"]

    def process(self):
        logging.info("Processing drawio final node prediction")


class DrawIOPrintNode(PredictionBase):
    prediction_class: Literal["print-node"]

    def process(self):
        logging.info("Processing drawio print node prediction")


class DrawIOVariableNode(PredictionBase):
    prediction_class: Literal["variable-node"]

    def process(self):
        logging.info("Processing drawio variable node prediction")


class DrawIOOperationNode(PredictionBase):
    prediction_class: Literal["operation-node"]

    def process(self):
        logging.info("Processing drawio operation node prediction")


class DrawIODecisionNode(PredictionBase):
    prediction_class: Literal["desicion-node"]

    def process(self):
        logging.info("Processing drawio decision node prediction")


class DrawIOFunctionNode(PredictionBase):
    prediction_class: Literal["function-node"]

    def process(self):
        logging.info("Processing drawio function node prediction")


class DrawIODecisionArrow(PredictionBase):
    prediction_class: Literal["desicion-arrow"]

    def process(self):
        logging.info("Processing drawio decision arrow prediction")


class DrawIONormalArrow(PredictionBase):
    prediction_class: Literal["normal-arrow"]

    def process(self):
        logging.info("Processing drawio normal arrow prediction")
