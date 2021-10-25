import numpy as np
import numba as nb
from data_structures import PixelSet, sparse_type

@nb.njit
def project(voxel_set: sparse_type,
            eu: nb.float64[:],
            ev: nb.float64[:]) -> sparse_type:
    '''
    Project voxel coordinates to two arbitrary axes, return
    a set of pixels and the voxel-to-pixel mapping

    Args:
        voxel_set (dict): Sparse set of voxels
        eu (int)        : Coordinates of the first unit vector
        ev (int)        : Coordinates of the second unit vecor
    Returns:
        dict : Sparse set of projected pixels
        array: (N) List of projected pixel IDs (one per voxel)
    '''
    # First check that the axes form an normed orthogonal basis
    assert (np.linalg.norm(eu)-1.) < 1e-9 and (np.linalg.norm(ev)-1.) < 1e-9
    assert np.dot(eu, ev) < 1e-9

    # Project all voxel coordinates onto the two requested axes
    A = np.vstack((eu, ev))
    projs  = A @ voxel_set.centers.T

    # Bin converted values
    limits = (A @ voxel_set.vertices.T)
    ranges = np.array([[np.min(limits[0]), np.max(limits[0])],
                       [np.min(limits[1]), np.max(limits[1])]])
    dimensions = ranges[:,1] - ranges[:,0]
    bins = ((dimensions/voxel_set.dimensions[0])*voxel_set.bins[0]).astype(np.int64) # Arbitrary
    pixel_set = PixelSet(ranges, bins)
    values = voxel_set.values
    pixel_ids = np.empty(voxel_set.size, dtype=np.int64)
    for i, p in enumerate(projs.T):
        pixel_ids[i] = pixel_set.coords_to_id([p[0], p[1]])
        pixel_set.add(pixel_ids[i], values[i])

    # Convert bin ids into an order list of key index
    for i, idx in enumerate(pixel_ids):
        pixel_ids[i] = pixel_set.index(idx)

    return pixel_set, pixel_ids

@nb.njit
def project_to_base_axis(voxel_set: sparse_type,
                         ai: nb.int64,
                         aj: nb.int64) -> sparse_type:
    '''
    Projects 3D voxels onto a pair of base axis

    Args:
        voxel_set (dict): Sparse set of voxels
        ai (int)        : ID of the first axis
        aj (int)        : ID of the second axis
    Returns:
        dict : Sparse set of projected pixels
        array: (N) List of projected pixel IDs (one per voxel)
    '''
    # Narrow coordinates down to requested axes only
    ranges = np.vstack((voxel_set.ranges[ai], voxel_set.ranges[aj]))
    bins = np.array([voxel_set.bins[ai],voxel_set.bins[aj]], dtype=np.int64)
    pixel_set = PixelSet(ranges, bins)
    pixel_ids = np.empty(voxel_set.size, dtype=np.int64)
    values = voxel_set.values
    for i, v in enumerate(voxel_set.positions):
        pixel_ids[i] = pixel_set.pos_to_id([int(v[ai]), int(v[aj])])
        pixel_set.add(pixel_ids[i], values[i])

    # Convert bin ids into an order list of key index
    for i, idx in enumerate(pixel_ids):
        pixel_ids[i] = pixel_set.index(idx)

    return pixel_set, pixel_ids
