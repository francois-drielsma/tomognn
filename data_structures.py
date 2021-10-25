import numpy as np
import numba as nb

sparse_type = [
    ('_dim', nb.int64),
    ('_ranges', nb.float64[:,:]),
    ('_bins', nb.int64[:]),
    ('_dict', nb.types.DictType(nb.int64, nb.float64)),
]

@nb.experimental.jitclass(sparse_type)
class SparseSet:
    '''
    Class which stores a sparse set of binned data and contains routines
    to convert bin position to coordinates and vice versa

    Attributes:
        _dim (int)     : Number of dimensions d
        _ranges (array): (d,2) array of volume boundaries
        _bins   (array): (d) array of number of bins in each dimension
        _dict (dict)   : Dictionary of (bin_id, value) pairs
    '''
    def __init__(self: sparse_type,
                 dim: nb.int64,
                 ranges: nb.float64[:,:],
                 bins: nb.int64[:]):
        '''
        Initialize object

        Args:
            dim (int)     : Number of dimensions d
            ranges (array): (d,2) array of volume boundaries
            bins   (array): (d) array of number of bins in each dimension
        '''
        # Assert that the input fit the assumptions
        assert ranges.shape == (dim,2), 'Need to provide lower and upper limits for each dimension'
        assert (ranges[:,0] < ranges[:,1]).all() , 'Specify the lower bound of the range first'
        assert len(bins) == dim, 'Need to provide a number of bins for each dimension'
        assert (bins > 0).all(), 'Need at least 1 bin in each dimension'

        # Save the input as attributes, instantiate dictionary (backend)
        self._dim = dim
        self._ranges = ranges
        self._bins   = bins
        self._dict   = nb.typed.Dict.empty(nb.int64, nb.float64)

    def index(self: sparse_type,
              idx: nb.int64) -> nb.int64[:]:
        '''
        Returns the position of a key in the ordered dictionary

        Args:
            idx (int): Bin ID
        Returns:
            array: Index of the key in the ordered list
        '''
        return list(self._dict.keys()).index(idx)

    def pos(self: sparse_type,
            idx: nb.int64) -> nb.int64[:]:
        '''
        Returns the position of a bin given its global ID

        Args:
            idx (int): Bin ID
        Returns:
            array: Bin position in each axis
        '''
        if idx < 0 or idx > np.prod(self._bins):
            raise ValueError('Bin ID outside of bounds')
        pos = np.empty(self._dim, np.int64)
        for i in range(self._dim):
            pos[i] = idx//np.prod(self._bins[i+1:])
            idx -= pos[i]*np.prod(self._bins[i+1:])

        return pos

    def center(self: sparse_type,
               idx: nb.int64) -> nb.float64[:]:
        '''
        Returns the coordinates of a bin center, given its global bin ID

        Args:
            idx (int): Bin ID
        Returns:
            array: Bin center in the original space
        '''
        if idx < 0 or idx > np.prod(self._bins):
            raise ValueError('Bin ID outside of bounds')
        return self._ranges[:,0] + (self.pos(idx) + 0.5)*self.dimensions/self._bins

    def llim(self: sparse_type,
             idx: nb.int64) -> nb.float64[:]:
        '''
        Returns the coordinates of a bin lower limits

        Args:
            idx (int): Bin ID
        Returns:
            array: Bin lower limit in the original space
        '''
        return self._ranges[:,0] + (self.pos(idx))*self.dimensions/self._bins

    def ulim(self: sparse_type,
             idx: nb.int64) -> nb.float64[:]:
        '''
        Returns the coordinates of a bin upper limit

        Args:
            idx (int): Bin ID
        Returns:
            array: Bin upper limit in the original space
        '''
        return self._ranges[:,0] + (self.pos(idx)+1)*self.dimensions/self._bins

    def value(self: sparse_type,
              idx: nb.int64) -> nb.float64[:]:
        '''
        Returns the value of a bin

        Args:
            idx (int): Bin ID
        Returns:
            float: Value
        '''
        return self._dict[idx]

    def add(self: sparse_type,
            idx: nb.int64,
            val: nb.float64 = 1.):
        '''
        Adds a bin to the SparseSet (increments if bin exist)

        Args:
            idx (int)  : Bin ID
            val (float): Value to increment by
        '''
        self._dict[idx] = self._dict[idx] + val if idx in self._dict else val

    def replace(self: sparse_type,
                idx: nb.int64,
                val: nb.float64 = 1.):
        '''
        Replaces a bin in the SparseSet (create if it does not exist)

        Args:
            idx (int)  : Bin ID
            val (float): Value to replace with
        '''
        self._dict[idx] = val

    def pos_to_id(self: sparse_type,
                  pos: nb.int64[:]) -> nb.int64:
        '''
        Returns the global ID of a bin, given its bin position

        Args:
            pos (array): Position of the bin
        Returns:
            int: Bin ID
        '''
        assert len(pos) == self._dim
        binid = 0
        for i in range(self._dim):
            if pos[i] < 0 or pos[i] > self._bins[0]-1:
                raise ValueError('Bin position outside of bounds')
            binid += pos[i]*np.prod(self._bins[i+1:])

        return binid

    def coords_to_id(self: sparse_type,
                     coords: nb.float64[:]) -> nb.int64:
        '''
        Returns the global ID of the bin a point falls into

        Args:
            coords (array): Coordinates of the point
        Returns:
            int: Bin ID
        '''
        assert len(coords) == self._dim
        binid = 0
        for i in range(self._dim):
            if coords[i] < self._ranges[i,0] or coords[i] > self._ranges[i,1]:
                raise ValueError('Point coordinates outside of bounds')
            pos = int(self._bins[i]*(coords[i]-self._ranges[i,0])/self.dimensions[i])
            binid += pos*np.prod(self._bins[i+1:])

        return binid

    def add_from_pos(self: sparse_type,
                     array: nb.float64[:]):
        '''
        Adds a bin to the SparseSet from bin pos + value (increments if bin exist)

        Args:
            array (float): List of bin [pos, value]
        '''
        assert len(array) == self._dim + 1
        self.add(self.pos_to_id(array[:self._dim]), array[-1])

    def add_from_coords(self: sparse_type,
                        array: nb.float64[:]):
        '''
        Adds a bin to the SparseSet from point coordinates + value (increments if bin exist)

        Args:
            array (float): List of point [coords, value]
        '''
        assert len(array) == self._dim + 1
        self.add(self.coords_to_id(array[:self._dim]), array[-1])

    def replace_from_pos(self: sparse_type,
                         array: nb.float64[:]):
        '''
        Replaces a bin in the SparseSet from bin pos + value (create if it does not exist)

        Args:
            array (float): List of bin [pos, value]
        '''
        assert len(array) == self._dim + 1
        self.replace(self.pos_to_id(array[:self._dim]), array[-1])

    def replace_from_coords(self: sparse_type,
                            array: nb.float64[:]):
        '''
        Replaces a bin in the SparseSet from point coordinates + value (create if it does not exist)

        Args:
            array (float): List of point [coords, value]
        '''
        assert len(array) == self._dim + 1
        self.replace(self.coords_to_id(array[:self._dim]), array[-1])

    def import_tensor(self: sparse_type,
                      tensor: nb.float64[:,:]):
        '''
        Loads a tensor of [coords,vals] into the set
        '''
        for row in tensor:
            self.add_from_coords(row)

    @property
    def ranges(self: sparse_type) -> nb.float64[:,:]:
        '''
        Returns the volume ranges
        '''
        return self._ranges

    @property
    def bins(self: sparse_type) -> nb.int64[:]:
        '''
        Returns the number of bins in each axis
        '''
        return self._bins

    @property
    def size(self: sparse_type) -> nb.int64:
        '''
        Returns the number of active bins (nonzero value)
        '''
        return len(self._dict)

    @property
    def dimensions(self: sparse_type) -> nb.float64[:]:
        '''
        Returns the dimensions of the image
        '''
        return self._ranges[:,1] - self._ranges[:,0]

    @property
    def vertices(self: sparse_type) -> nb.float64[:,:]:
        '''
        Returns the 8 vertices which bound the image
        '''
        mask = ((np.arange(2**self._dim).reshape(-1,1) & (1 << np.arange(self._dim))) > 0)
        return ~mask*self._ranges[:,0] + mask*self._ranges[:,1]

    @property
    def bin_size(self: sparse_type) -> nb.float64[:]:
        '''
        Returns the dimensions of a bin
        '''
        return self.dimensions/self._bins

    @property
    def positions(self: sparse_type) -> nb.int64[:,:]:
        '''
        Return list of bin positions
        '''
        positions = np.empty((len(self._dict), self._dim), dtype=np.int64)
        for i, idx in enumerate(self._dict.keys()):
            positions[i] = self.pos(idx)
        return positions

    @property
    def values(self: sparse_type) -> nb.float64[:]:
        '''
        Return list of bin values
        '''
        values = np.empty(len(self._dict), dtype=np.float64)
        for i, v in enumerate(self._dict.values()):
            values[i] = v
        return values

    @property
    def centers(self: sparse_type) -> nb.float64[:,:]:
        '''
        Return list of bin centers in the original space
        '''
        centers = np.empty((len(self._dict), self._dim), dtype=np.float64)
        for i, idx in enumerate(self._dict.keys()):
            centers[i] = self.center(idx)
        return centers

    @property
    def lower_limits(self: sparse_type) -> nb.float64[:,:]:
        '''
        Return list of bin lower limits in the original space
        '''
        llims = np.empty((len(self._dict), self._dim), dtype=np.float64)
        for i, idx in enumerate(self._dict.keys()):
            llims[i] = self.llim(idx)
        return llims

    @property
    def upper_limits(self: sparse_type) -> nb.float64[:,:]:
        '''
        Return list of bin upper limits in the original space
        '''
        ulims = np.empty((len(self._dict), self._dim), dtype=np.float64)
        for i, idx in enumerate(self._dict.keys()):
            ulims[i] = self.ulim(idx)
        return ulims

    @property
    def tensor(self: sparse_type) -> nb.float64[:,:]:
        '''
        Returns the sparse set in the form of a (N,d+1) tensor of [coords,val]
        '''
        tensor = np.empty((len(self._dict), self._dim+1), dtype=np.float64)
        for i, item in enumerate(self._dict.items()):
            tensor[i,:self._dim] = self.center(item[0])
            tensor[i,-1]         = item[1]

        return tensor


@nb.njit
def PixelSet(ranges: nb.float64[:,:],
             bins: nb.int64[:]) -> sparse_type:
    '''
    Alias for a sparse set of dimension 2

    Args:
        ranges (array): (2,2) array of image boundaries
        bins   (array): (2) array of number of bins in each dimension
    '''
    return SparseSet(2, ranges, bins)


@nb.njit
def VoxelSet(ranges: nb.float64[:,:],
             bins: nb.int64[:]) -> sparse_type:
    '''
    Alias for a sparse set of dimension 3

    Args:
        ranges (array): (3,2) array of image boundaries
        bins   (array): (3) array of number of bins in each dimension
    '''
    return SparseSet(3, ranges, bins)
