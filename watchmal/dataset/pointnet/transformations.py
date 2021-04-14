def x_flip(data):
    data[0,:] = -data[0, :]
    return data

def y_flip(data):
    data[1,:] = -data[1, :]
    return data

def z_flip(data):
    data[2,:] = -data[2, :]
    return data