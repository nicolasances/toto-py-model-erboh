
from remote.totoml_registry import check_registry

from processes.training import TrainingProcess
from processes.scoring import ScoreProcess
from processes.predict_single import predict as predict_single
from processes.predict_batch import predict as predict_batch

class ModelController: 

    def __init__(self, model_name): 

        # Init variables
        self.model_name = model_name;

        # Check if the model exists on the registry. If it does not, create it.
        check_registry(self.model_name)

    def train(self, request): 
        """
        Retrains the model 
        """
        TrainingProcess(self.model_name, request).do()

        return {"success": True, "message": "Model {} trained successfully".format(self.model_name)}

    def score(self, request): 
        """
        Calculate the accuracy (metrics) of the champion model
        """
        metrics = ScoreProcess(self.model_name, request).do()

        return {"metrics": metrics}

    def predict_single(self, message):
        """
        Predicts on a single item
        """
        predict_single(message)
    
    def predict_batch(self, message):
        """
        Predicts on a batch of items
        """
        predict_batch(message)


