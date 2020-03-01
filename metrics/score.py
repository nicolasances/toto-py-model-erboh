from sklearn.metrics import classification_report, confusion_matrix, f1_score
from toto_logger.logger import TotoLogger

logger = TotoLogger();

class Scorer: 

    def __init__(self, cid): 
        self.correlation_id = cid
    
    def do(self, y, y_pred): 

        f1 = f1_score(y, y_pred)

        logger.compute(self.correlation_id, '[ SCORING ] - F1 score: {}'.format(f1),'info')

        return [
            {"name": "F1", "value": f1}
        ]