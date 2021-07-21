"""
Transformation functions used for data augmentation
"""

from torch import flip
import torch
import numpy as np

__all__ = ['horizontal_flip', 'vertical_flip', ]

horizontal_flip_mpmt_map=[0, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 12, 17, 16, 15, 14, 13, 18]
vertical_flip_mpmt_map=[6, 5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 15, 14, 13, 12, 17, 16, 18]


def horizontal_flip(data):
    return flip(data[horizontal_flip_mpmt_map, :, :, ], [2])

def vertical_flip(data):
    return flip(data[vertical_flip_mpmt_map, :, :], [1])


def mpmtPadding(data, barrel_rows):
    half_len_index = int(data.shape[2]/2)
    horiz_pad_data = torch.cat((data, torch.zeros_like(data[:, :, :half_len_index])), 2)
    horiz_pad_data[:, :, 2*half_len_index:] = torch.tensor(0, dtype=torch.float64)
    horiz_pad_data[:, barrel_rows, 2*half_len_index:] = data[:, barrel_rows, :half_len_index]
    
    l_index = data.shape[2]/2 - 1
    r_index = data.shape[2]/2
    
    l_endcap_ind = int(l_index - 4)
    r_endcap_ind = int(r_index + 4)
    
    top_end_cap = data[:, barrel_rows[-1]+1:, l_endcap_ind:r_endcap_ind+1]
    bot_end_cap = data[:, :barrel_rows[0], l_endcap_ind:r_endcap_ind+1]
    
    vhflip_top = horizontal_flip(vertical_flip(top_end_cap))
    vhflip_bot = horizontal_flip(vertical_flip(bot_end_cap))
    
    horiz_pad_data[:, barrel_rows[-1]+1:, l_endcap_ind + int(data.shape[2]/2) : r_endcap_ind + int(data.shape[2]/2) + 1] = vhflip_top
    horiz_pad_data[:, :barrel_rows[0], l_endcap_ind + int(data.shape[2]/2) : r_endcap_ind + int(data.shape[2]/2) + 1] = vhflip_bot
    
    return horiz_pad_data
