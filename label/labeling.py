
def data_labeled(event): 
    '''
    Reacts to data (expense) being labeled by the user. 
    This event is received when a user has CHANGED a "monthly" label.

    When this happens, erboh will update a "counter" of predictions having been changed, in order to then trigger a 
    retraining when that counter reaches a treshold.

    NOTE: how do we manage to calculate the model accuracy? 
    '''
    
    pass