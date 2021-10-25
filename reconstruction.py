import numpy as np
import numba as nb

def combine_projections(projs, bases, tolerance):
    '''
    Combine projections into 3D spacepoints
    - Compare the first two projections to form candidate space points
    - Cross-check against the remaining axes
    - Position of a spacepoint is the mean estimate from all projections
    - Value of a spacepoint is the mean value from all projections

    Args:
        projs (list(array)): List of projection matrices, each containing (x, x_i, val)
        bases (list(array)): List of projection axis pairs
        tolerance (double) : Max allowed distance (in bins) between associated hits
    Returns:
        array: Matrix of reconstructed space points (x,y,z,val)
    '''
    return _combine_projections(nb.typed.List(projs), nb.typed.List(bases), tolerance)

@nb.njit
def _combine_projections(projs: nb.typed.List(),
                         bases: nb.types.List(nb.float64[:,:]),
                         tolerance: np.float64 = 1) -> nb.float64[:,:]:

    # Extract the coordinates and values from the projections
    coords, values = [], []
    for i, p in enumerate(projs):
        coords.append(np.ascontiguousarray(p.centers.T))
        values.append(p.values)

    # Each projection restricts the position of the 3D point to a ray that is
    # perpendicular to the projection plane: find normals to the projection planes
    normals = np.empty((len(bases),3), dtype=np.float64)
    for i, base in enumerate(bases):
        normals[i] = np.cross(base[0], base[1])

    # Find the points where the rays interecept the projection planes
    intercepts = []
    for i, c in enumerate(coords):
        intercepts.append(c.T @ bases[i])

    # Find the doublets of points compatible across the first two projections
    bin_size = projs[0].bin_size[0] # Assume constant bin size across all axes, TODO
    dist01, spacepoints = lines_to_lines_separation(intercepts[0], intercepts[1], normals[0], normals[1])
    mask01 = dist01 < tolerance*bin_size
    doublets = np.vstack(np.where(mask01))

    # Each doublet forms a unique 3d spacepoint
    spacepoints = spacepoints.reshape(-1,3)[mask01.flatten()]
    charges     = (values[0][doublets[0]]+values[1][doublets[1]])/2

    # Now cross-check these spacepoints against the remaining projections, remove if necessary
    mask = np.ones(len(spacepoints), dtype=np.bool_)
    for k in range(2, len(coords)):
        dist0k, _  = lines_to_lines_separation(intercepts[0], intercepts[k], normals[0], normals[k])
        for i in range(len(spacepoints)):
            neighborhood = np.where(dist0k[doublets[0][i]] < tolerance*bin_size)[0]
            ints = intercepts[k][neighborhood]
            meanpos, meanval, nfound = np.zeros(3, dtype=ints.dtype), 0., 0
            for j in range(len(ints)):
                dist = np.linalg.norm(np.cross(ints[j] - spacepoints[i], normals[k]))
                if dist < tolerance*bin_size:
                    meanpos += (ints[j] + (spacepoints[i]-ints[j]).dot(normals[k])*normals[k])
                    meanval += values[k][neighborhood[j]]
                    nfound  += 1

            mask[i] = bool(nfound)
            if nfound:
                spacepoints[i] = (k*spacepoints[i] + meanpos/nfound)/(k+1)
                charges[i] = (k*charges[i] + meanval/nfound)/(k+1)

    return np.hstack((spacepoints[mask], charges[mask].reshape(-1,1)))


def combine_projections_shared_axis(projs, bases, tolerance):
    '''
    Here we leverage the knowledge that the first axis is shared across all projections
    - Compare the first two projections to form candidate space points
    - Cross-check against the remaining axes
    - Position of a spacepoint is the mean estimate from all projections
    - Value of a spacepoint is the mean value from all projections

    Args:
        projs (list(array)): List of projected PixelSets
        bases (list(array)): List of projection axis pairs
        tolerance (double) : Max allowed distance (in bins) between associated hits
    Returns:
        array: Matrix of reconstructed space points (x,y,z,val)
    '''
    return _combine_projections_shared_axis(nb.typed.List(projs), nb.typed.List(bases), tolerance)

@nb.njit
def _combine_projections_shared_axis(projs: nb.typed.List(),
                                     bases: nb.types.List(nb.float64[:,:]),
                                     tolerance: np.float64 = 1) -> nb.float64[:,:]:

    # Check that the first axis is the same in all projections
    assert shared_axis(bases), "First axis must be shared across all projections"

    # Extract the coordinates and values from the projections
    coords, values = [], []
    for i, p in enumerate(projs):
        coords.append(np.ascontiguousarray(p.centers.T))
        values.append(p.values)

    # Find the doublets of points compatible across the first two projections
    bin_size = projs[0].bin_size[0] # Assume constant bin size across all axes, TODO
    dist01   = np.abs(coords[1][0]-coords[0][0].reshape(-1,1))
    doublets = np.vstack(np.where(dist01 < tolerance*bin_size))

    # Each doublet forms a unique 3d spacepoint, find its coordinates
    common      = (coords[0][0][doublets[0]]+coords[1][0][doublets[1]])/2
    starts      = common.reshape(-1,1)*bases[0][0]
    projections = np.vstack((coords[0][1][doublets[0]], coords[1][1][doublets[1]])).T
    base        = np.vstack((bases[0][1], bases[1][1]))
    spacepoints = starts + coplanar_points(projections, base)
    charges     = (values[0][doublets[0]]+values[1][doublets[1]])/2

    # Now cross-check these spacepoints against the remaining projections, remove if necessary
    mask = np.ones(len(spacepoints), dtype=np.bool_)
    for k in range(2, len(coords)):
        dist0k     = np.abs(coords[k][0]-coords[0][0].reshape(-1,1))
        intercepts = coords[k].T @ bases[k]
        normal     = np.cross(bases[k][0], bases[k][1])
        for i in range(len(spacepoints)):
            neighborhood = np.where(dist0k[doublets[0][i]] < tolerance*bin_size)[0]
            ints = intercepts[neighborhood]
            meanpos, meanval, nfound = np.zeros(3, dtype=ints.dtype), 0., 0
            for j in range(len(ints)):
                dist = np.linalg.norm(np.cross(ints[j] - spacepoints[i], normal))
                if dist < tolerance*bin_size:
                    meanpos += (ints[j] + (spacepoints[i]-ints[j]).dot(normal)*normal)
                    meanval += values[k][neighborhood[j]]
                    nfound  += 1

            mask[i] = bool(nfound)
            if nfound:
                spacepoints[i] = (k*spacepoints[i] + meanpos/nfound)/(k+1)
                charges[i] = (k*charges[i] + meanval/nfound)/(k+1)

    return np.hstack((spacepoints[mask], charges[mask].reshape(-1,1)))


@nb.njit
def coplanar_point(projections: nb.float64[:],
                   base: nb.float64[:,:]) -> nb.float64[:]:
    '''
    Reconstructs the 3d point that is coplanar to the base
    axes onto which it is projected

    Args:
        projections (array): (2) Value of the projection on each axis
        base (array)       : (2,3) Base projection axes
    Returns:
        array: (3) Coordinates of the point
    '''
    # First, find the normal (n) to the two base axes
    normal = np.cross(base[0], base[1])

    # Solver Cramer system where x.e_u = u, x.e_v = v and x.n = 0
    A = np.vstack((base, normal.reshape(1,-1)))
    p = np.linalg.inv(A) @ np.array([projections[0], projections[1], 0.], dtype=np.float64)
    return p


@nb.njit
def coplanar_points(projections: nb.float64[:,:],
                    base: nb.float64[:,:]) -> nb.float64[:]:
    '''
    Reconstructs 3d points that are coplanar to the base
    axes onto which they are projected

    Args:
        projections (array): (N,2) Value of the projections on each axis
        base (array)       : (2,3) Base projection axes
    Returns:
        array: (N,3) Coordinates of the points
    '''
    # First, find the normal (n) to the two base axes
    normal = np.cross(base[0], base[1])

    # Solve Cramer system where x.e_u = u, x.e_v = v and x.n = 0
    A      = np.vstack((base, normal.reshape(1,-1)))
    Ainv   = np.linalg.inv(A)
    points = np.empty((len(projections), 3), dtype = projections.dtype)
    rhs    = np.hstack((projections, np.zeros((len(projections),1), dtype=projections.dtype)))
    for i in range(len(rhs)):
        points[i] = Ainv @ rhs[i]

    return points


@nb.njit
def line_to_line_separation(points: nb.float64[:,:],
                            directions: nb.float64[:,:]) -> (nb.float64[:], nb.float64[:]):
    '''
    Algorithm that finds the closest points of approach of two sets of lines and
    returns their pairwise distance and midpoint

    Args:
        points (array)    : (2,3) array of line start points
        directions (array): (2,3) array of line directions (*normalized*)
    Returns:
        array: closest points of approach
    '''
    # Get the angle between vectors
    d = points[0] - points[1]
    vi, vj = directions[0], directions[1]
    v_dp = np.dot(vi, vj)

    # Minimize the distance
    si = (-np.dot(d,vi) + np.dot(d,vj)*v_dp)/(1-v_dp**2)
    sj = ( np.dot(d,vj) - np.dot(d,vi)*v_dp)/(1-v_dp**2)

    # Minimum separation
    cpai = points[0] + si * directions[0]
    cpaj = points[1] + sj * directions[1]

    return np.linalg.norm(cpaj-cpai), (cpai+cpaj)/2


@nb.njit
def lines_to_lines_separation(Xi: nb.float64[:,:],
                              Xj: nb.float64[:,:],
                              diri: nb.float64[:],
                              dirj: nb.float64[:]) -> (nb.float64[:], nb.float64[:,:]):
    '''
    Algorithm that finds the closest points of approach of two sets of lines and
    returns their pairwise distances and midpoints
    - Assumes lines that belong in the same batch share a common direction

    Args:
        Xi (array)  : (N,3) array of line intercept points in the first batch
        Xj (array)  : (M,3) array of line interecept points in the second batch
        diri (array): (1,3) Line direction in the first batch
        dirj (array): (1,3) Line direction in the second batch
    '''
    # Compute the projections of the point displacement onto the
    Ai, Aj = np.empty((len(Xi), len(Xj)), dtype=Xi.dtype), np.empty((len(Xi), len(Xj)), dtype=Xi.dtype)
    for i, xi in enumerate(Xi):
        for j, xj in enumerate(Xj):
            Ai[i,j], Aj[i,j] = (xi-xj).dot(diri), (xi-xj).dot(dirj)

    # Compute the offsets for the closest points of approach
    v_dp = np.dot(diri, dirj)
    si = (-Ai + Aj*v_dp)/(1.-v_dp**2)
    sj = ( Aj - Ai*v_dp)/(1.-v_dp**2)

    # Compute the closest points of approach, record distance and midpoints
    cpai = Xi.reshape(len(Xi),1,3) + si.reshape(*si.shape,1)*diri
    cpaj = Xj.reshape(1,len(Xj),3) + sj.reshape(*sj.shape,1)*dirj

    return np.sqrt(np.sum((cpaj-cpai)**2, axis=-1)), (cpai+cpaj)/2


@nb.njit
def shared_axis(bases):
    '''
    Function which returns true if one of the projection axes is shared

    Args:
        bases (list(array)): List of projection axis pairs
    Returns:
        bool: True if the first axis is shared
    '''
    for i in range(len(bases)-1):
        for j in range(i+1,len(bases)):
            if not (bases[i][0] == bases[j][0]).all():
                return False
    return True


@nb.njit
def norm_nb(X: nb.float64[:,:],
            axis: nb.int64) -> nb.float64[:]:
    '''
    Numba norm does not support axis option, add it

    Args:
        X (array) : (N,M) 2d array of values
        axis (int): axis along which to take the norm
    Returns:
        array: (N/M) array of norms
    '''
    return np.sqrt(np.sum(X**2), axis=axis)


@nb.njit
def mean_nb(X: nb.float64[:,:],
            axis: nb.int64) -> nb.float64[:]:
    '''
    Numba mean does not support axis option, add it

    Args:
        X (array) : (N,M) 2d array of values
        axis (int): axis along which to take the mean
    Returns:
        array: (N/M) array of means
    '''
    return np.sum(X, axis=axis)/X.shape[axis]
