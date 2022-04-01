def x_flip(data):
    data[0,:] = -data[0, :]
    if data.shape[0] > 6:
        data[3] = -data[3, :]
    return data

def y_flip(data):
    data[1,:] = -data[1, :]
    if data.shape[0] > 6:
        data[4] = -data[4, :]
    return data

def z_flip(data):
    data[2,:] = -data[2, :]
    if data.shape[0] > 6:
        data[5] = -data[5, :]
    return data