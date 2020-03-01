from .remote import check_registry, create_retrained_model
from .gcpremote import save_retrained_model_pickle

class ModelController: 

    def __init__(self, model_name): 

        # Init variables
        self.model_name = model_name;

        # Check if the model exists on the registry. If it does not, create it.
        check_registry(self.model_name)

    def post_retrained_model(self, data, cid): 
        """
        Post a new retrained model to the current champion model

        Parameters
        ----------
        data : object
            The data object must contain the following fields: 
            - model : the model object
            - metrics : an array [] of metrics. Each metric is an object containing: 
             - name : string, the name of the metric
             - value : anything, the value of the metric
        """
        # 1. Create a new retrained model and post the metrics
        create_retrained_model(self.model_name, data['metrics'], cid);

        # 2. TODO: save the model of the challenger
        save_retrained_model_pickle(self.model_name, data['model'])
