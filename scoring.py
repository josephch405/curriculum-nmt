def get_transfer_scores(order_name, dataset):
    """ Returns the transfer learning scores given a pretrained network.
    """
    if order_name == "bert": 

def rank_scores(transfer_values):
    """ Ranks the transfer learning scores.
    """
    pass

def load_order(order_name, dataset):
    """ Takes in a dataset and returns an ordering (ranking) 
    based on a scoring (difficulty) function.
    @param order_name (str): the scoring function to use
    @param dataset (Dataset): dataset to rank
    @returns order (list): an ordering of ranked scores
    """
    networks = ["bert"]
    if order_name not in networks:
        raise ValueError("Order name {} is not supported!".format(order_name))
    # Get transfer learning scores
    transfer_values = get_transfer_scores(order_name=order_name, dataset)
    order = rank_scores(transfer_values)
    
def balance_order(order, dataset):
    """ Ensures that the ordered dataset according to transfer learning 
    difficulty is balanced, i.e. each class has equal number of samples.
    """
    pass
