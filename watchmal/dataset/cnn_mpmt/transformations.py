from torch import flip

__all__ = ['horizontal_flip', 'vertical_flip', ]

horizontal_flip_mpmt_map=[0, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 12, 17, 16, 15, 14, 13, 18]
vertical_flip_mpmt_map=[6, 5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 15, 14, 13, 12, 17, 16, 18]


def horizontal_flip(data):
    return flip(data[horizontal_flip_mpmt_map, :, :, ], [2])

def vertical_flip(data):
    return flip(data[vertical_flip_mpmt_map, :, :], [1])