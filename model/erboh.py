# from processes.predict_single import SinglePredictor
# from processes.predict_batch import BatchPredictor
# from processes.training import TrainingProcess
# from processes.scoring import ScoreProcess
from totoml.delegate import ModelDelegate
from totoml.model import ModelType

from model.train import TrainingProcess
from model.predict import SinglePredictor
from model.predict_batch import BatchPredictor
from model.score import ScoreProcess

from toto_logger.logger import TotoLogger

logger = TotoLogger()

class ERBOH: 

    def __init__(self):
        pass

    def get_model_type(self):
        return ModelType.sklearn

    def get_name(self): 
        return "erboh"

    def predict(self, model, context, data): 

        return SinglePredictor().predict(model, context, data)

    def predict_batch(self, model, context, data=None): 

        return BatchPredictor().predict(model, context, data)

    def train(self, model_info, context):

        return TrainingProcess().train(model_info, context)

    def score(self, model, context):

        return ScoreProcess().score(model, context)        

        