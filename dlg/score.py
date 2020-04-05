import pandas as pd 
from sklearn.metrics import precision_recall_fscore_support, f1_score

from toto_logger.logger import TotoLogger

logger = TotoLogger();

class Scorer: 

    def __init__(self, cid, context=''): 
        self.correlation_id = cid
        self.context = context
    
    def do(self, y, y_pred): 

        accuracy = precision_recall_fscore_support(y, y_pred)

        # Prepare data frame
        # This model is mostly interested in :
        # - Recall on class 1: did I manage to catch all 1s (all actual monthly expenses)
        # – Precision on class 1: did I manage to not wrongly classify stuff as "monthly"
        acc_df = pd.DataFrame(accuracy, columns=['Class 0', 'Class 1'], index=['Precision', 'Recall', 'F1', 'Support'])

        f1 = f1_score(y, y_pred)

        logger.compute(self.correlation_id, '[ {context} ] - [ SCORING ] - Precision Class 1: {v}'.format(context=self.context, v=acc_df.loc['Precision', 'Class 1']),'info')
        logger.compute(self.correlation_id, '[ {context} ] - [ SCORING ] - Recall Class 1: {v}'.format(context=self.context, v=acc_df.loc['Recall', 'Class 1']),'info')
        logger.compute(self.correlation_id, '[ {context} ] - [ SCORING ] - F1 score: {v}'.format(context=self.context, v=f1),'info')

        return [
            {"name": "Precision Class 1", "value": acc_df.loc['Precision', 'Class 1']},
            {"name": "Recall Class 1", "value": acc_df.loc['Recall', 'Class 1']},
            {"name": "F1 score", "value": f1}
        ]