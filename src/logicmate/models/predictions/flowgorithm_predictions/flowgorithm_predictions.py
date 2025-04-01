import logging

from typing import Literal
from logicmate.models.predictions.predictions.prediction import PredictionBase


class FlowgorithmInitialNode(PredictionBase):
    prediction_class: Literal["initial-node"]
    text: Literal["Inicio"]

    def process(self):
        logging.info("Processing flowgorithm initial node prediction")


class FlowgorithmFinalNode(PredictionBase):
    prediction_class: Literal["final-node"]
    text: Literal["Fin"]

    def process(self):
        logging.info("Processing flowgorithm final node prediction")


class FlowgorithmInputNode(PredictionBase):
    prediction_class: Literal["input-node"]

    def process(self):
        logging.info("Processing flowgorithm input node prediction")


class FlowgorithmOutputNode(PredictionBase):
    prediction_class: Literal["output-node"]

    def process(self):
        logging.info("Processing flowgorithm output node prediction")


class FlowgorithmOperationNode(PredictionBase):
    prediction_class: Literal["operation-node"]

    def process(self):
        logging.info("Processing flowgorithm operation node prediction")


class FlowgorithmDecisionNode(PredictionBase):
    prediction_class: Literal["decision-node"]

    def process(self):
        logging.info("Processing flowgorithm decision node prediction")


class FlowgorithmNormalArrow(PredictionBase):
    prediction_class: Literal["normal-arrow"]

    def process(self):
        logging.info("Processing flowgorithm normal arrow prediction")


class FlowgorithmDecisionArrow(PredictionBase):
    prediction_class: Literal["decision-arrow"]

    def process(self):
        logging.info("Processing flowgorithm decision arrow prediction")
