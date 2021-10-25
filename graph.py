import numpy as np
import numba as nb
from data_structures import sparse_type
from transform import project
from reconstruction import shared_axis, line_to_line_separation, lines_to_lines_separation

def build_spacepoints(pixel_sets, edge_index, bases):
    '''
    Uses edge predictions to build spacepoints

    Args:
        pixel_sets (list(dict)): List of sparse sets of projected pixels
        edge_index (array)     : List of activated edges
        bases (list(array))    : List of projection axis pairs
    Returns:
        array: Value and coordinates
    '''
    return _build_spacepoints(nb.typed.List(pixel_sets), edge_index, nb.typed.List(bases))

@nb.njit
def _build_spacepoints(pixel_sets: nb.typed.List(),
                       edge_index: nb.int64[:,:],
                       bases: nb.types.List(nb.float64[:,:])) -> nb.float64[:,:]:

    # Convert edge index using unique IDs for each node
    n_projs = len(pixel_sets)
    sizes, offsets = np.empty(n_projs, dtype=np.int64), np.zeros(n_projs+1, dtype=np.int64)
    for i in range(n_projs):
        sizes[i]       = pixel_sets[i].size
        offsets[i+1:] += sizes[i]
    n_nodes = np.sum(sizes)
    edge_index_combined = np.empty((len(edge_index), 2), dtype=np.int64)
    for i, e in enumerate(edge_index):
        edge_index_combined[i,0] = offsets[e[0]]+e[2]
        edge_index_combined[i,1] = offsets[e[1]]+e[3]

    # Loop over the nodes in the first projection
    degrees = np.zeros(n_nodes, dtype=np.int64)
    combinations = []
    for i in range(sizes[0]):
        # Find the neighbors of the node
        maski = np.where(edge_index_combined[:,0] == i)[0]
        neighborsi = edge_index_combined[maski, 1]
        projs = edge_index[maski,1]
        for j in neighborsi[projs==1]:
            maskj = np.where(edge_index_combined[:,0] == j)[0]
            neighborsj = edge_index_combined[maskj, 1]
            for k in neighborsi[projs==2]:
                if (neighborsj == k).any():
                    combinations.append(np.array([i,j,k])-offsets[:-1])
                    degrees[i] += 1
                    degrees[j] += 1
                    degrees[k] += 1

    # Each projection restricts the position of the 3D point to a ray that is
    # perpendicular to the projection plane: find normals to the projection planes
    normals = np.empty((len(bases),3), dtype=np.float64)
    for i, base in enumerate(bases):
        normals[i] = np.cross(base[0], base[1])

    # Find the points where the rays interecept the projection planes
    intercepts, values = [], []
    for i, ps in enumerate(pixel_sets):
        intercepts.append(ps.centers @ bases[i])
        values.append(pixel_sets[i].values)

    points = np.empty((len(combinations),n_projs+1), dtype=np.float64)
    for k, c in enumerate(combinations):
        point = np.zeros(3, dtype=np.float64)
        for i in range(n_projs):
            for j in range(i+1, n_projs):
                point += line_to_line_separation(np.vstack((intercepts[i][c[i]], intercepts[j][c[j]])),
                                                 np.vstack((normals[i], normals[j])))[1]
        value = 0
        for i in range(n_projs):
            value += values[i][c[i]]/degrees[offsets[i]+c[i]]

        points[k,:3] = point/3
        points[k,-1] = value/3

    return points

def graph_input(voxel_set, bases, tolerance):
    '''
    Graph representation of the tomographic reconstruction problem

    Args:
        voxel_set (dict)   : Sparse set of voxels
        bases (list(array)): List of projection axis pairs
        tolerance (float)  : Max allowed distance between associated hits
    Returns:
        list(dict) : List of sparse sets of projected pixels
        list(array): List of (N_i,12) node features, one per projection
        array      : (E,4) array of edges [P_i, P_j, i, j]
        array      : (E) boolean array of edge labels (True if valid, False otherwise)
    '''
    return _graph_input(voxel_set, nb.typed.List(bases), tolerance)

@nb.njit
def _graph_input(voxel_set: sparse_type,
                 bases: nb.types.List(nb.float64[:,:]),
                 tolerance: nb.float64):

    # Project voxels onto the requested planes
    pixel_sets, pixel_ids = [], []
    for i, b in enumerate(bases):
        proj, ids = project(voxel_set, b[0], b[1])
        pixel_sets.append(proj)
        pixel_ids.append(ids)

    # Get the node features
    node_features = _projected_point_features(pixel_sets, bases)

    # Get the edge index
    coords = [p.centers for p in pixel_sets]
    if shared_axis(bases):
        edge_index = _compatible_pairs_shared_axis(coords, bases, tolerance*pixel_sets[0].bin_size[0])
    else:
        edge_index = _compatible_pairs(coords, bases, tolerance*pixel_sets[0].bin_size[0])

    # Get the edge labels
    edge_labels = _get_edge_labels(edge_index, pixel_sets, pixel_ids)

    return pixel_sets, node_features, edge_index, edge_labels


def projected_point_features(projs, bases):
    '''
    Transform each pixel in a projection into a graph node with features

    Args:
        projs (list(dict)) : List of projection sets
        bases (list(array)): List of projection axis pairs
    Retunrs:
        List: List of array of node features (one per projection)
    '''
    return _projected_point_features(nb.typed.List(projs), nb.typed.List(bases))

@nb.njit
def _projected_point_features(projs, bases):
    node_features = []
    for i in range(len(projs)):
        features = np.empty((projs[i].size, 12), dtype=np.float32)
        normal   = np.cross(bases[i][0], bases[i][1])
        coords, values = projs[i].centers, projs[i].values
        for j in range(projs[i].size):
            features[j, :2]  = coords[j]
            features[j, 2]   = values[j]
            features[j,3:6]  = bases[i][0]
            features[j,6:9]  = bases[i][1]
            features[j,9:12] = normal

        node_features.append(features)

    return node_features


def compatible_pairs(coords, bases, tolerance):
    '''
    Function which generates an edge index (i.e. sparse adjacency matrix)
    which joins compatible points from separate projections

    Args:
        coords (list(array)): List of projected coordinates
        bases (list(array)) : List of projection axis pairs
        tolerance (float)  : Max allowed distance between associated hits
    Returns:
        array: (E,4) array of edges [P_i, P_j, i, j]
    '''

    return _compatible_pairs(nb.typed.List(coords), nb.typed.List(bases), tolerance)

@nb.njit
def _compatible_pairs(coords: nb.types.List(nb.float64[:,:]),
                      bases: nb.types.List(nb.float64[:,:]),
                      tolerance: np.float64) -> nb.int64[:,:]:

    # Find the intecepts on of the projection rays with the projection planes and the plane normals
    n_projs = len(bases)
    intercepts, normals = [], []
    for i in range(n_projs):
        intercepts.append(coords[i] @ bases[i])
        normals.append(np.cross(bases[i][0], bases[i][1]))

    # Find the points which are within the tolerance in each pairs of projections
    pairs = np.empty((0,4), dtype = np.int64)
    for i in range(n_projs):
        for j in range(i+1, n_projs):
            distij = lines_to_lines_separation(intercepts[i], intercepts[j], normals[i], normals[j])[0]
            maskij = np.vstack(np.where(distij < tolerance)).T
            projij = np.vstack((i*np.ones(len(maskij), dtype=np.int64), j*np.ones(len(maskij), dtype=np.int64))).T
            pairs  = np.vstack((pairs, np.hstack((projij, maskij))))

    return pairs


def compatible_pairs_shared_axis(coords, bases, tolerance):
    '''
    Function which generates an edge index (i.e. sparse adjacency matrix)
    which joins compatible points from separate projections

    Here we leverage the knowledge that the first axis is shared across all projections

    Args:
        coords (list(array)): List of projected coordinates
        bases (list(array)) : List of projection axis pairs
        tolerance (double)  : Max allowed distance between associated hits
    Returns:
        array: (E,4) array of edges [P_i, P_j, i, j]
    '''

    return _compatible_pairs_shared_axis(nb.typed.List(coords), nb.typed.List(bases), tolerance)

@nb.njit
def _compatible_pairs_shared_axis(coords: nb.types.List(nb.float64[:,:]),
                                  bases: nb.types.List(nb.float64[:,:]),
                                  tolerance: np.float64) -> nb.int64[:,:]:
    # Check that the first axis is the same in all projections
    assert shared_axis(bases), "First axis must be shared across all projections"

    # Find the points which are within the tolerance in each pairs of projections
    n_projs = len(coords)
    pairs   = np.empty((0,4), dtype=np.int64)
    for i in range(n_projs):
        for j in range(i+1, n_projs):
            distij = np.abs(np.ascontiguousarray(coords[j].T)[0]-np.ascontiguousarray(coords[i].T)[0].reshape(-1,1))
            maskij = np.vstack(np.where(distij < tolerance)).T
            projij = np.vstack((i*np.ones(len(maskij), dtype=np.int64), j*np.ones(len(maskij), dtype=np.int64))).T
            pairs  = np.vstack((pairs, np.hstack((projij, maskij))))

    return pairs


def get_edge_labels(edge_index, pixel_sets, pixel_ids):
    '''
    Returns the labels of the predicted edges by cross-checking them
    against the valid combinations

    Args:
        edge_index (array)     : (E,4) Array of edges between projected points [P_i,P_j,i,j]
        pixel_sets (list(dict)): List of sparse sets of pixels
        pixel_ids (list(array)): List of pixel ids corresponding to true space points
    Returns:
        array: (E) boolean array of edge labels (True if valid, False otherwise)
    '''
    return _get_edge_labels(edge_index, nb.typed.List(pixel_sets), nb.typed.List(pixel_ids))

@nb.njit
def _get_edge_labels(edge_index: nb.int64[:,:],
                     pixel_sets: nb.typed.List(),
                     pixel_ids: nb.types.List(nb.int64[:])) -> nb.bool_[:]:
    valid = np.empty((3, len(pixel_ids[0])), dtype=np.int64)
    for i in range(len(pixel_ids)):
        valid[i] = pixel_ids[i]
    edge_labels = np.zeros(len(edge_index), dtype=np.bool_)
    for i, e in enumerate(edge_index):
        if ((valid[e[0]] == e[2]) & (valid[e[1]] == e[3])).any():
            edge_labels[i] = True
    return edge_labels
