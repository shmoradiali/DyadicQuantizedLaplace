import numpy as np


# Distributed Mean Estimation

def scalar_dme_fixed_x(x, k, enc, dec):
    """
    Perform the DME experiment for a fixed value.
    enc: The encoding mechanism (client)
    dec: The decoding mechanism (server)
    x: The constant value each user holds
    k: The number of transmissions (clients)
    """
    mse = 0
    avg_comm = 0
    for _ in range(k):
        m = enc.encode(x, output_binary=True)
        x_hat = dec.decode(m, input_binary=True)
        
        mse += (x_hat - x)**2
        avg_comm += len(m)
    
    avg_comm /= k
    mse /= k
    return mse, avg_comm


def scalar_dme_unif_x(k, enc, dec):
    """
    Perform the DME experiment for a X ~ Unif(-1, 1).
    enc: The encoding mechanism (client)
    dec: The decoding mechanism (server)
    x: The constant value each user holds
    k: The number of transmissions (clients)
    """
    mse = 0
    avg_comm = 0
    for _ in range(k):
        x = np.random.random() * 2 - 1
        m = enc.encode(x, output_binary=True)
        x_hat = dec.decode(m, input_binary=True)
    
        mse += (x_hat - x)**2
        avg_comm += len(m)

    avg_comm /= k
    mse /= k 

    return mse, avg_comm


def scalar_dme_list(xs, k, enc, dec):
    """
    Perform the DME experiment for each value in xs.
    enc: The encoding mechanism (client)
    dec: The decoding mechanism (server)
    xs: The list of values
    k: The number of transmissions (clients)
    """

    avg_mse = 0
    avg_comm_cost = 0
    
    for x in xs:
        mse, comm_cost = scalar_dme_fixed_x(x, k, enc, dec)
        avg_mse += mse
        avg_comm_cost += comm_cost
    
    avg_mse /= xs.size
    avg_comm_cost /= xs.size

    return avg_mse, avg_comm_cost
    