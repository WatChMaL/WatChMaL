import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# DEFINE CONSTANTS FOR SHORT TANK GEOMETRY

barrel_half_height = 296.4490661621094
barrel_radius = 399.0980529785156
R = barrel_radius

endcap_limit = 290.

# offset for endcaps due to flattening
y_offset = barrel_half_height + barrel_radius

# offsets when making plot assume positive values only
positive_x_offset = np.pi*barrel_radius
lower_endcap_offset = y_offset + barrel_radius
upper_endcap_offset = y_offset + lower_endcap_offset

# set up dimensions for preimage with short tank data
min_pmt_x_value = 17
max_pmt_x_value = 2496

min_pmt_y_value = 101
max_pmt_y_value = 2099


preimage_dimensions = [max_pmt_y_value + min_pmt_y_value + 1, max_pmt_x_value + min_pmt_x_value + 1]

# ========================================================================
barrel_map_array_idxs = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 15, 16, 17, 12, 13, 14, 18]
pmts_per_mpmt = 19

def get_event_data_from_index(item, all_hit_pmts, all_hit_data, event_hits_index, data_size, mpmt_positions):
    start = event_hits_index[item]
    stop  = event_hits_index[item + 1]

    hit_pmts = all_hit_pmts[start:stop].astype(np.int16)
    hit_data = all_hit_data[start:stop]

    hit_mpmts = hit_pmts // pmts_per_mpmt
    hit_pmt_in_modules = hit_pmts % pmts_per_mpmt

    hit_rows = mpmt_positions[hit_mpmts, 0]
    hit_cols = mpmt_positions[hit_mpmts, 1]

    sample_data = np.zeros(data_size)
    sample_data[hit_pmt_in_modules, hit_rows, hit_cols] = hit_data

    # fix barrel array indexing to match endcaps in xyz ordering
    sample_data[:, 12:28, :] = sample_data[barrel_map_array_idxs, 12:28, :]
    
    return hit_pmts, sample_data[hit_pmt_in_modules, hit_rows, hit_cols].flatten() 

# ========================================================================
# Mapping and plotting functions

def PMT_to_flat_cylinder_mapping( tubes, tube_xyz ):
    """
    Build dictionary of PMT number, to (x,y) on a flat cylinder
    
    N.B. Tube numbers in full geometry file go from 1:NPMTs, but it seems like
    the event data number from 0:NPMTs-1, so subtracting 1 from tube number here?
    """
    mapping = {}
    for idx, tube in enumerate(tubes):
        x = tube_xyz[idx,0]
        y = tube_xyz[idx,1]
        z = tube_xyz[idx,2]
        if ( y > endcap_limit ):
            # in top circle of cylinder
            xflat = x
            yflat = y_offset + z
            mapping[ int( tube-1 ) ] = [ xflat, yflat ]
            
        elif ( y < -endcap_limit):
            # in bottom circle of cylinder+
            xflat = x
            yflat = -y_offset + z
            mapping[ int( tube-1 ) ] = [ xflat, yflat ]
            
        else:
            # in barrel part of cylinder
            theta = math.atan2( z, x )
            xflat = R * theta
            yflat = y
            mapping[ int( tube-1 ) ] = [ xflat, yflat ]
    return mapping

def PMT_to_flat_cylinder_map_positive( tubes, tube_xyz ):
    """
    Build dictionary of PMT number, to (x,y) on a flat cylinder
    
    N.B. Tube numbers in full geometry file go from 1:NPMTs, but it seems like
    the event data number from 0:NPMTs-1, so subtracting 1 from tube number here?
    
    """
    mapping = {}
    for idx, tube in enumerate(tubes):
        x = tube_xyz[idx,0]
        y = tube_xyz[idx,1]
        z = tube_xyz[idx,2]
        if ( y > endcap_limit ):
            # in top circle of cylinder
            xflat = x + positive_x_offset
            yflat = z + upper_endcap_offset
            mapping[ int( tube-1 ) ] = [ int(round(xflat)), int(round(yflat)) ]
            
        elif ( y < -endcap_limit):
            # in bottom circle of cylinder
            xflat = x + positive_x_offset
            yflat = z + barrel_radius
            mapping[ int( tube-1 ) ] = [ int(round(xflat)), int(round(yflat)) ]
            
        else:
            # in barrel part of cylinder
            theta = math.atan2( z, x )
            xflat = R * theta + np.pi*barrel_radius
            yflat = y + lower_endcap_offset
            mapping[ int( tube-1 ) ] = [ int(round(xflat)), int(round(yflat)) ]
    return mapping

def EventDisplay( tubes, quantities, PMTFlatMapPositive, title="Charge", cutrange=[-1,-1] ):
    """
    tubes == np.array of PMTs that were hit
    quantities == np.array of PMT quantities (either charge or time)
    title == title to add to display
    cutrange == minimum and maximum values on plot (or set both same for default)
    """
    
    fig, ax= plt.subplots(figsize=[30,30])
    preimage = np.zeros( preimage_dimensions )
    
    imgmin = quantities.min()
    imgmax = quantities.max()

    for idx, tube in enumerate( tubes ):
        if cutrange[0] != cutrange[1]:
            if quantities[idx] < cutrange[0] or quantities[idx] > cutrange[1]:
                continue
        for dx in range(-3,4):
            for dy in range(-3,4):
                if abs(dx)==3 and abs(dy)==3:
                    continue
                    
                #print( "idx=", idx, " len(quantities)=",len(quantities), " tube=", tube, " len(PMTFlatMap)=", len(PMTFlatMapPositive))
                preimage[ PMTFlatMapPositive[tube][1]+dx, PMTFlatMapPositive[tube][0]+dy ] = quantities[idx]

    if cutrange[0] != cutrange[1]:
        imgmin = cutrange[0]
        imgmax = cutrange[1]
    
    im = ax.imshow( preimage, extent = [-positive_x_offset,positive_x_offset,-lower_endcap_offset,lower_endcap_offset], vmin=imgmin, vmax=imgmax )

    fig.suptitle(title, fontsize=80)

    plt.rc('xtick', labelsize=24) 
    plt.rc('ytick', labelsize=24) 
    plt.xlabel('Distance CCW on perimeter from x-axis (cm)', fontsize=48)
    plt.ylabel('Y (cm)', fontsize=48)

    plt.set_cmap('gist_heat_r')

    # Create colourbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=24)

    # Fix title height
    plt.subplots_adjust(top=0.5)
    plt.tight_layout()

# ========================================================================
# Subset mapping and plotting functions

def get_upper_endcap_tubes( tubes, tube_xyz ):
    """
    returns ids of PMTs lying in the upper endcap
    """
    subset_tubes = []
    for idx, tube in enumerate(tubes):
        x = tube_xyz[idx,0]
        y = tube_xyz[idx,1]
        z = tube_xyz[idx,2]
        if ( y > endcap_limit ):
            # in upper endcap
            subset_tubes.append(tube)
    return np.array(subset_tubes)

def get_lower_endcap_tubes( tubes, tube_xyz ):
    """
    returns ids of PMTs lying in the lower endcap
    """
    subset_tubes = []
    for idx, tube in enumerate(tubes):
        x = tube_xyz[idx,0]
        y = tube_xyz[idx,1]
        z = tube_xyz[idx,2]
        if ( y < -endcap_limit ):
            # in lower endcap
            subset_tubes.append(tube)
    return np.array(subset_tubes)

def get_barrel_tubes( tubes, tube_xyz ):
    """
    returns ids of PMTs lying in the barrel
    """
    subset_tubes = []
    for idx, tube in enumerate(tubes):
        x = tube_xyz[idx,0]
        y = tube_xyz[idx,1]
        z = tube_xyz[idx,2]
        if ( -endcap_limit < y < endcap_limit ):
            # in barrel part of cylinder
            subset_tubes.append(tube)
    return np.array(subset_tubes)

def EventSubsetDisplay( tubes, quantities, PMTFlatMapPositive, tubes_to_plot, title="Charge", cutrange=[-1,-1], padding=10):
    """
    tubes == np.array of PMTs that were hit
    quantities == np.array of PMT quantities (either charge or time)
    title == title to add to display
    cutrange == minimum and maximum values on plot (or set both same for default)
    """
    PMTFlatMapPositive_values = [PMTFlatMapPositive[tube] for tube in tubes_to_plot]
    subset_x_values = np.array([value[0] for value in PMTFlatMapPositive_values])
    subset_y_values = np.array([value[1] for value in PMTFlatMapPositive_values])
    
    # set up dimensions for subset preimage with short tank data
    min_subplot_x_value = subset_x_values.min() - padding
    max_subplot_x_value = subset_x_values.max() + padding

    min_subplot_y_value = subset_y_values.min() - padding
    max_subplot_y_value = subset_y_values.max() + padding
    
    fig, ax= plt.subplots(figsize=[30,30])
    preimage = np.zeros( preimage_dimensions )

    subset_quantities = []
    for idx, tube in enumerate( tubes ):
        if cutrange[0] != cutrange[1]:
            if quantities[idx] < cutrange[0] or quantities[idx] > cutrange[1]:
                continue
        for dx in range(-3,4):
            for dy in range(-3,4):
                if abs(dx)==3 and abs(dy)==3:
                    continue
                if tube in tubes_to_plot:    
                    #print( "idx=", idx, " len(quantities)=",len(quantities), " tube=", tube, " len(PMTFlatMap)=", len(PMTFlatMapPositive))
                    preimage[ PMTFlatMapPositive[tube][1]+dx, PMTFlatMapPositive[tube][0]+dy ] = quantities[idx]
                    subset_quantities.append(quantities[idx])
    
    subset_quantities = np.array(subset_quantities)

    imgmin = subset_quantities.min()
    imgmax = subset_quantities.max()
    
    if cutrange[0] != cutrange[1]:
        imgmin = cutrange[0]
        imgmax = cutrange[1]
    
    subset_image = preimage[min_subplot_y_value:max_subplot_y_value, min_subplot_x_value:max_subplot_x_value]
    
    im = ax.imshow( subset_image, extent = [min_subplot_x_value, max_subplot_x_value, min_subplot_y_value, max_subplot_y_value], vmin=imgmin, vmax=imgmax )

    fig.suptitle(title, fontsize=80)

    plt.rc('xtick', labelsize=24) 
    plt.rc('ytick', labelsize=24) 
    plt.xlabel('Distance CCW on perimeter from x-axis (cm)', fontsize=48)
    plt.ylabel('Y (cm)', fontsize=48)
    
    plt.set_cmap('gist_heat_r')

    # Create colourbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=24)

    # Fix title height
    plt.tight_layout()