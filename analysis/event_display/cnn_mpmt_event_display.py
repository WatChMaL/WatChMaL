"""
Tools for event displays from CNN mPMT dataset
"""
import numpy as np
import analysis.event_display.event_display as ed
from watchmal.dataset.cnn_mpmt.cnn_mpmt_dataset import CNNmPMTDataset


def channel_position_offset(channel):
    channel = channel % 19
    theta = (channel<12)*2*np.pi*channel/12 + ((channel >= 12) & (channel<18))*2*np.pi*(channel-12)/6
    radius = 0.2*(channel<18) + 0.2*(channel<12)
    position = [radius*np.cos(theta), radius*np.sin(theta)] # note this is [y, x] or [row, column]
    return position


def positions_from_data(data):
    indices = np.indices(data.shape)
    channels = indices[0].flatten()
    positions = indices[1:].reshape(2,-1).astype(np.float64)
    positions += channel_position_offset(channels)
    return positions


def plot_2d_data(data, mpmt_pos, old_convention=False, **kwargs):
    positions = positions_from_data(data)
    if old_convention:
        positions[1] = max(mpmt_pos[:, 1])-positions[1]
    data_nan = np.full_like(data, np.nan)
    data_nan[:, mpmt_pos[:, 0], mpmt_pos[:, 1]] = data[:, mpmt_pos[:, 0], mpmt_pos[:, 1]]
    ed.plot_2d_event(data_nan, positions, mpmt_pos, **kwargs)


def plot_2d_event(dataset, event, **kwargs):
    data = dataset.__getitem__(event)['data']
    plot_2d_data(data, dataset.mpmt_positions, **kwargs)


def plot_3d_event(dataset, event, geo_positions, **kwargs):
    dataset.__getitem__(event)
    hit_positions = geo_positions[dataset.event_hit_pmts, :]
    data = np.column_stack((hit_positions, dataset.event_hit_charges))
    ed.plot_3d_event(data, geo_positions, **kwargs)


CNNmPMTDataset.plot_2d_event = plot_2d_event
CNNmPMTDataset.plot_3d_event = plot_3d_event
