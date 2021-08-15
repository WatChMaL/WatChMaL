import numpy as np


def towall(position, angle, tank_half_height=300, tank_radius=400, tank_axis=1):
    """
    Calculate towall: distance from position to detector wall, in particle direction

    Parameters
    ----------
    position : array_like
        vector of (x, y, z) position of a particle or (N,3) array of positions for N particles
    angle : array_like
        vector of (zenith, azimuth) direction of particle or (N, 2) array of directions for N particles
    tank_half_height : float, default: 300
        half-height of the detector ID
    tank_radius : float, default: 400
        Radius of the detector ID
    tank_axis : int, default: 1
        axis along which the tank cylinder is oriented
    """
    pos_trans = np.delete(position, tank_axis, axis=-1)
    pos_along = position[..., tank_axis]
    zenith = angle[..., 0]
    azimuth = angle[..., 1]
    dir_along = np.cos(zenith)
    dir_trans = np.column_stack((np.sin(zenith)*np.cos(azimuth), np.sin(zenith)*np.sin(azimuth)))
    a = np.linalg.norm(dir_trans, axis=-1)**2
    b = np.sum(pos_trans*dir_trans, axis=-1)
    c = np.linalg.norm(pos_trans, axis=-1) ** 2 - tank_radius ** 2
    towall_barrel = (-b + np.sqrt(b**2-a*c)) / a
    towall_endcap = tank_half_height / abs(dir_along) - pos_along / dir_along
    return np.minimum(towall_barrel, towall_endcap)


def dwall(position, tank_half_height=300, tank_radius=400, tank_axis=1):
    """
        Calculate dwall: distance from position to nearest detector wall

        Parameters
        ----------
        position : array_like
            vector of (x, y, z) position of an event or (N,3) array of (x, y, z) position coordinates for N events
        tank_half_height : float, default: 300
            half-height of the detector ID
        tank_radius : float, default: 400
            Radius of the detector ID
        tank_axis : int, default: 1
            axis along which the tank cylinder is oriented
        """
    pos_along = position[..., tank_axis]
    pos_trans = np.delete(position, tank_axis, axis=-1)
    dwall_barrel = tank_radius - np.linalg.norm(pos_trans, axis=-1)
    dwall_endcap = tank_half_height - np.abs(pos_along)
    return np.minimum(dwall_barrel, dwall_endcap)


def momentum(energy, label, particle_masses=np.array((0, 0.511, 105.7, 134.98))):
    """
        Calculate momentum of particle from total energy and particle type (label)
        Default labels are 0:gamma, 1:electron, 2:muon, 3:pi0

        Parameters
        ----------
        energy : array_like
            energy of particle or vector of energies of particles
        label : array_like
            integer label of particle type or vector of labels of particles
        particle_masses : array_like
            array of particle masses indexed by label
        """
    mass = particle_masses[label]
    return np.sqrt(energy**2 - mass**2)
