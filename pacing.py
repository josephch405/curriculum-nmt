import numpy as np

# [ ---- PACING FUNCTIONS ---- ]


def none():
    return 1


def linear(time, warmup_iters, slope, bias):
    """ Simple linear pacing function that takes in a slope and a bias 
    and computes the pacing over time. If not slope is None, then compute 
    slope based on the max time.
    @param time (int): current train iter
    @param max_epoch (int): max epoch
    @param n_iters (int): number of iterations per epoch
    @param slope (int): step size per unit of time
    @param bias (int): initial value at t=0 in (0,1)
    @returns pacing (int): pacing value in (0,1)
    """
    if slope is None:
        slope = 1 / warmup_iters
    return min([1, slope * time + bias])


def root(time, warmup_iters, bias, p=2):
    """ Root pacing function.
    @param time (int): epoch
    @param max_epoch (int): max epoch
    @param n_iters (int): number of iterations per epoch
    @param bias (int): bias in (0,1)
    @param p (int): root-p sharpness 
    @returns pacing (int): pacing value in (0,1)
    """
    return min([1, np.sqrt(time * ((1 - bias**2) / warmup_iters) + bias**2)])

# ------------------------------


def pacing_data(train_data, dev_data, time=None, warmup_iters=5000, method=None, tb=None):
    """ Gets the data according to the specified pacing function.
    @param train_data (list): list of tuple of sents
    @param dev_data (list): list of tuple of sents
    @param time (int): current train iteration
    @param max_epoch (int): max epoch
    @param n_iters (int): number of iterations per epoch
    @param method (str): desired pacing function
    @returns paced_dataset (Dataset): dataset of (current_train_data, current_dev_data)
    """
    pacing_functions = ["none", "linear", "root"]
    if method not in pacing_functions:
        raise ValueError("Pacing function {} is not supported!".format(method))

    if method == "none":
        pacing = none()
    elif method == "linear":
        pacing = linear(
            time=time,
            warmup_iters=warmup_iters,
            slope=None,
            bias=0.05
        )
        # print("linear pacing is:", pacing)
    elif method == "root":
        pacing = root(time=time,
                      warmup_iters=warmup_iters, bias=0.1)
        # print("root pacing is:", pacing)

    # Slice dataset according to pacing
    pacing_train_idx = int(pacing * len(train_data))
    pacing_dev_idx = int(pacing * len(dev_data))
    sliced_train = train_data[:pacing_train_idx]
    sliced_dev = dev_data[:pacing_dev_idx]

    if tb:
        tb.add_scalar('pacing', pacing, time)
        tb.add_scalar('pacing_idx', pacing_train_idx, time)

    paced_dataset = (sliced_train, sliced_dev)
    return paced_dataset


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # TODO: refactor to get rid of niters
    print("Creating plot for pacing functions.")
    T = 30
    n_iters = 100
    x = list(range(T*n_iters))

    # None
    y = [none() for t in x]
    plt.plot(x, y)

    # Linear
    y = [linear(t, T, n_iters, slope=None, bias=0.01) for t in x]
    plt.plot(x, y)

    # Root
    y = [root(t, T, n_iters, bias=0.01) for t in x]
    plt.plot(x, y)

    plt.legend(['none', 'linear', 'root'], loc='upper left')
    plt.xlabel("Time")
    plt.ylabel("Pacing")
    save_path = 'figures/pacing_functions.png'
    plt.savefig(save_path)
    print("Figure saved to {}".format(save_path))
    plt.close()
