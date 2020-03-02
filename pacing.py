import numpy as np

### [ ---- PACING FUNCTIONS ---- ] 

def linear(time, max_time, slope, bias):
    """ Simple linear pacing function that takes in a slope and a bias 
    and computes the pacing over time. If not slope is None, then compute 
    slope based on the max time.
    @param time (int): epoch
    @param max_time (int): max epoch
    @param slope (int): step size per unit of time
    @param bias (int): initial value at t=0
    @returns pacing (int): pacing value in (0,1)
    """
    if slope is None:
        slope = 1 / max_time
    return min([1, slope * time + bias])

def root(time, max_time, bias, p=2):
    """ Root pacing function.
    @param time (int): epoch
    @param max_time (int): max epoch
    @param bias (int): bias
    @param p (int): root-p sharpness 
    @returns pacing (int): pacing value in (0,1)
    """
    return min([1, np.sqrt(time * ((1 - bias**2) / max_time) + bias**2)])

### ------------------------------

def pacing_data(train_data, dev_data, time=None, max_time=None, method=None):
    """ Gets the data according to the specified pacing function.
    @param train_data (list): list of tuple of sents
    @param dev_data (list): list of tuple of sents
    @param time (int): epoch 
    @param max_time (int): max epoch
    @param method (str): desired pacing function
    @returns paced_dataset (Dataset): dataset of (current_train_data, current_dev_data)
    """
    pacing_functions = ["linear", "root"]
    if method not in pacing_functions:
        raise ValueError("Pacing function {} is not supported!".format(method))
    
    if method == "linear":
        pacing = linear(time=time, max_time=max_time, slope=None, bias=0.1)
        print("linear pacing is:", pacing)
    elif method == "root":
        pacing = root(time=time, max_time=max_time, bias=0.1)
        print("root pacing is:", pacing)

    # Slice dataset according to pacing
    pacing_train_idx = int(pacing * len(train_data))
    pacing_dev_idx = int(pacing * len(dev_data))
    sliced_train = train_data[:pacing_train_idx]
    sliced_dev = dev_data[:pacing_dev_idx]
    
    paced_dataset = (sliced_train, sliced_dev)
    return paced_dataset
