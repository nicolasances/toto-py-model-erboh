from dlg.storage import FileStorage
from toto_pubsub.publisher import TotoEventPublisher
from toto_logger.logger import TotoLogger

logger = TotoLogger()

publisher = TotoEventPublisher(microservice='model-erboh', topics=['erboh-train'])

def data_labeled(event): 
    '''
    Reacts to data (expense) being labeled by the user. 
    This event is received when a user has CHANGED a "monthly" label.

    When this happens, erboh will update a "counter" of predictions having been changed, in order to then trigger a 
    retraining when that counter reaches a treshold.

    The event must contain:
    - user (string): the email of the user
    - id (string): the expense id
    - monthly (boolean): the monthly flag (updated by the user)
    '''
    # Get the data out
    user = event['user']
    cid = event['correlationId']
    monthly = event['monthly']
    expense_id = event['id']

    if monthly == True: 
        monthly = 1
    else: 
        monthly = 0

    # TODO: the model name and versions have to be taken from somewhere else
    storage = FileStorage('model-erboh', 1)

    # 1. Update the predictions file and recalculate accuracy
    accuracy_df = storage.save_label_and_accuracy({"id": expense_id, "monthly": monthly}, user)

    prec_class_1 = accuracy_df.loc['Precision', 'Class 1']
    reca_class_1 = accuracy_df.loc['Recall', 'Class 1']

    logger.compute(cid, 'New model ({model} - v{version}) accuracy: Precision on 1: {prec}, Recall on 1: {reca}'.format(model='model-erboh', version=1, prec=prec_class_1, reca=reca_class_1), 'info')

    # 2. If the accuracy has gone under a treshold, retrain
    # TODO: parameterize or rule that checks whether the accuracy has dropped x points from previous accuracy
    threshold_reca_class1 = 0.9
    threshold_prec_class1 = 0.8

    retrain = False
    
    # We're checking ONLY the recall or precision on class 1
    if (prec_class_1 < threshold_prec_class1) or (reca_class_1 < threshold_reca_class1): 
        logger.compute(cid, 'Accuracy under threshold', 'info')
        retrain = True

    # TODO: if the model is already retraining, nevermind

    if retrain: 
        logger.compute(cid, 'Requesting a new training of the model', 'info')
        # Post an event to trigger the retrain process
        retrain_event = {
            'correlationId': cid,
            'model': 'model-erboh',
            'version': 1,
            'recall_class_1': accuracy_df.loc['Recall', 'Class 1'],
            'precision_class_1': accuracy_df.loc['Precision', 'Class 1']
        }

        publisher.publish(topic='erboh-train', event=retrain_event)


# Example of event: {"correlationId": "201901021211021291112121", "user": "nicolas.matteazzi@gmail.com", "id": "5e2304aa2767f82b4e900648", "monthly": false}