import numpy as np

row_remap=np.flip(np.arange(16))

"""
The index starts at 1 and counts up continuously with no gaps
Each 19 consecutive PMTs belong to one mPMT module, so (index-1)/19 is the module number.
The index%19 gives the position in the module: 1-12 is the outer ring, 13-18 is inner ring, 0 is the centre PMT
The modules are then ordered as follows:
It starts by going round the second highest ring around the barrel, then the third highest ring, fourth highest
ring, all the way down to the lowest ring (i.e. skips the highest ring).
Then does the bottom end-cap, row by row (the first row has 6 modules, the second row has 8, then 10, 10, 10,
10, 10, 10, 8, 6).
Then the highest ring around the barrel that was skipped before, then the top end-cap, row by row.
I'm not sure why it has this somewhat strange order...
WTF: actually it is: 2, 6, 8 10, 10, 12 and down again in the caps
"""

def module_index(pmt_index):
    """Returns the module number given the 0-indexed pmt number"""
    return pmt_index//19

def pmt_in_module_id(pmt_index):
    """Returns the pmt number within a 
    module given the 0-indexed pmt number"""
    return pmt_index%19

def is_barrel(module_index):
    """Returns True if module is in the Barrel"""
    return ( (module_index<600) | ((module_index>=696)&(module_index<736)) )

def is_bottom(module_index):
    """Returns True if module is in the bottom cap"""
    return ( (module_index>=600)&(module_index<696) )

def is_top(module_index):
    """Returns True if module is in the top cap"""
    return ( (module_index>=736)&(module_index<832) )

def rearrange_barrel_indices(module_index):
    """rearrange indices to have consecutive module 
    indexing starting with top row in the barrel"""

    #check if there are non-barrel indices here
    is_not_barrel= ~is_barrel(module_index)
    any_not_barrel=np.bitwise_or.reduce(is_not_barrel)
    if any_not_barrel:
        raise ValueError('Passed a non-barrel PMT for geometry processing')
    
    rearranged_module_index=np.zeros_like(module_index)
    barrel_bulk_indices=np.where(module_index<600)
    #print "barrel_bulk_indices"
    #print barrel_bulk_indices
    barrel_top_row_indices=np.where( ((module_index>=696)&(module_index<736)) )
    #print "barrel_top_row_indices"
    #print barrel_top_row_indices
    rearranged_module_index[barrel_bulk_indices]=module_index[barrel_bulk_indices]+40
    rearranged_module_index[barrel_top_row_indices]=module_index[barrel_top_row_indices]-696
    #print "rearranged_module_index"
    #print rearranged_module_index
    return rearranged_module_index


def row_col_rearranged(rearranged_barrel_index):
    """return row and column index based on the rearranged module indices"""
    #print "rearanged_barrel_index: "
    #print rearranged_barrel_index
    #print "rearanged_barrel_index//40: "
    #print rearranged_barrel_index//40
    
    row=row_remap[rearranged_barrel_index//40]
    col=rearranged_barrel_index%40
    return row, col

def row_col(module_index):
    """return row and column from a raw module index"""
    #print "module_index:"
    #print module_index
    return row_col_rearranged(rearrange_barrel_indices(module_index))
