import watchmal.dataset.data_utils as du

def x_flip(data):
    """Returns point-cloud formatted data with the sign flipped of the x-component of the position and direction"""
    data[0, :] = -data[0, :]
    if data.shape[0] > 6:
        data[3] = -data[3, :]
    return data


def y_flip(data):
    """Returns point-cloud formatted data with the sign flipped of the y-component of the position and direction"""
    data[1, :] = -data[1, :]
    if data.shape[0] > 6:
        data[4] = -data[4, :]
    return data


def z_flip(data):
    """Returns point-cloud formatted data with the sign flipped of the z-component of the position and direction"""
    data[2, :] = -data[2, :]
    if data.shape[0] > 6:
        data[5] = -data[5, :]
    return data

def random_reflections(data):
    """Returns point-cloud formatted data with the position and direction randomly flipped in each axis"""
    return du.apply_random_transformations([x_flip, y_flip, z_flip], data)