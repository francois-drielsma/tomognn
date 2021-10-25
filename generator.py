import numpy as np
import numba as nb

from data_structures import VoxelSet, sparse_type

gen_type = [
    ('_ranges', nb.float64[:,:]),
    ('_bins', nb.int64[:])
]

@nb.experimental.jitclass(gen_type)
class TrackGenerator:
    '''
    Class which calls Numba routines that produce
    random tracks inside a box and voxelize them.
    '''

    def __init__(self: gen_type,
                 ranges: nb.float64[:,:],
                 bins: nb.int64[:] = np.empty(0, dtype=np.int64)):
        '''
        Initializes the cube in which to sample a track

        Args:
            ranges (array): (3,2) array of volume boundaries
            bins   (array): (3) array of number of bins in each dimension
        '''
        self._ranges = ranges
        self._bins = bins

    def get_point(self: gen_type) -> nb.float64[:]:
        '''
        Gets a random point inside the volume

        Returns:
            array: Coordinates of the point
        '''
        return self._ranges[:,0] + self.dimensions*np.random.rand(3)

    def get_face_point(self: gen_type,
                       axis: nb.int64 = -1,
                       side: nb.int64 = -1,
                       exclude_axis: nb.int64 = -1,
                       exclude_side: nb.int64 = -1) -> (nb.float64[:], nb.int64, nb.int64):
        '''
        Gets a random point on the boundaries of the volume

        Args:
            axis (int)        : Axis the first face to use is perpendicular to (-1: random)
            side (int)        : Side the first face to use is on (-1: random)
            exclude_axis (int): Axis the face not to use is perpendicular to (-1: random)
            exclude_axis (int): Side the face not to use is on (-1: random)
        Returns:
            array: Coordinates of the point
        '''
        assert axis < 0 or side < 0 or axis != exclude_axis or side != exclude_side
        if axis < 0:
            axis = np.random.randint(3)
        if side < 0:
            side = np.random.randint(2)
        while axis == exclude_axis and side == exclude_side:
            axis = np.random.randint(3)
            side = np.random.randint(2)

        center = 0.5*(np.ones(3) + (2*side - 1)*(np.arange(3)==axis))
        offset = (np.arange(3)!=axis)*(np.random.rand(3)-0.5)

        return self._ranges[:,0] + self.dimensions*(center + offset), axis, side

    def get_segment(self: gen_type) -> (nb.float64[:], nb.float64[:]):
        '''
        Gets a random segment inside the volume

        Returns:
            array: Coordinates of the start points
            array: Coordinates of the end points
        '''
        return self.get_point(), self.get_point()

    def get_line(self: gen_type,
                 start_axis: nb.int64,
                 start_side: nb.int64,
                 vector_method: bool = True) -> (nb.float64[:], nb.float64[:]):
        '''
        Gets a random through line inside a cube

        Args:
            start_axis (int)    : Axis the first face to use is perpendicular to (-1: random)
            start_side (int)    : Side the first face to use is on (-1: random)
            vector_method (bool): If true, generates lines by taking a random direction vector
        Returns:
            array: Coordinates of the start points
            array: Coordinates of the end points
        '''
        if start_axis > -1 or start_side > -1:
            start, fa, fs = self.get_face_point(axis=start_axis, side=start_side)
        else:
            start = self.get_point()

        if vector_method:
            vec   = np.array([np.random.normal() for _ in range(3)])
            vec  /= np.linalg.norm(vec)
            vertices = []
            tol = 1e-9
            for axis in range(3):
                for side in range(2):
                    xi = (self._ranges[axis,side]-start[axis])/vec[axis]
                    intc = start + xi*vec
                    if (intc >= self._ranges[:,0]-tol).all() & (intc <= self._ranges[:,1]+tol).all():
                        maskl, masku = intc < self._ranges[:,0] + tol, intc > self._ranges[:,1] - tol
                        intc[maskl] = self._ranges[maskl, 0]
                        intc[masku] = self._ranges[masku, 1]
                        vertices.append(intc)

            assert len(vertices) == 2
            vertices = np.vstack((vertices[0], vertices[1]))
            np.random.shuffle(vertices)

            start, end = vertices[0], vertices[1]
        else:
            end, _, __  = self.get_face_point(exclude_axis=fa, exclude_side=fs)

        return start, end

    def get_segments(self: gen_type,
                     n: nb.int64[:]) -> (nb.float64[:,:], nb.float64[:,:]):
        '''
        Get a set of random segments inside a cube

        Args:
            n (float): Number of lines to produces
        Returns:
            array: Coordinates of the start points
            array: Coordinates of the end points
        '''
        starts, ends = np.empty((n,3), dtype=np.float64), np.empty((n,3), dtype=np.float64)
        for i in range(n):
            starts[i], ends[i] = self.get_segment()
        return starts, ends

    def get_lines(self: gen_type,
                  n: nb.int64 = 1,
                  start_axis: nb.int64 = -1,
                  start_side: nb.int64 = -1,
                  vector_method: bool = True) -> (nb.float64[:,:], nb.float64[:,:]):
        '''
        Gets a set of random through lines inside a cube

        Args:
            n (int)             : Number of lines to produces
            start_axis (int)    : Axis the first face to use is perpendicular to (-1: random)
            start_side (int)    : Side the first face to use is on (-1: random)
            vector_method (bool): If true, generates lines by taking a random direction vector
        Returns:
            array: Coordinates of the start points
            array: Coordinates of the end points
        '''
        starts, ends = np.empty((n,3), dtype=np.float64), np.empty((n,3), dtype=np.float64)
        for i in range(n):
            starts[i], ends[i] = self.get_line(start_axis, start_side, vector_method)
        return starts, ends

    def get_segment_voxels(self: gen_type,
                           n: nb.int64 = 1) -> (nb.float64[:,:], nb.float64[:,:]):
        '''
        Generates segments an voxelizes them

        Args:
            n (int): Number of segments to produce
        Returns:
            np.ndarray: VoxelSet object
        '''
        voxel_set = VoxelSet(self._ranges, self._bins)
        starts, ends = self.get_segments(n)
        for i in range(n):
            self.append_track_voxels(voxel_set, [starts[i], ends[i]])
        return voxel_set, (starts, ends)

    def get_line_voxels(self: gen_type,
                        n: nb.int64 = 1,
                        start_axis: nb.int64 = -1,
                        start_side: nb.int64 = -1,
                        vector_method: bool = True) -> (nb.float64[:,:], nb.float64[:,:]):
        '''
        Generates lines an voxelizes them

        Args:
            n (int)             : Number of lines to produce
            start_axis (int)    : Axis the first face to use is perpendicular to (-1: random)
            start_side (int)    : Side the first face to use is on (-1: random)
            vector_method (bool): If true, generates lines by taking a random direction vector
        Returns:
            np.ndarray: VoxelSet object
        '''
        voxel_set = VoxelSet(self._ranges, self._bins)
        starts, ends = self.get_lines(n, start_axis, start_side, vector_method)
        for i in range(n):
            self.append_track_voxels(voxel_set, [starts[i], ends[i]])
        return voxel_set, (starts, ends)

    def append_track_voxels(self: gen_type,
                            voxel_set: sparse_type,
                            end_points: (nb.float64[:], nb.float64[:])):
        '''
        Converts a line (pair of end points) into a set of voxels

        Args:
            voxel_set (dict)  : Set of voxels to append
            end_points (array): End points of the line
        '''
        # Find line gradients
        start, end = end_points
        grads = end - start

        # Find the range of bins in the first axis, loop over them
        vox_size = voxel_set.bin_size
        s0, e0 = min(start[0], end[0]), max(start[0], end[0])
        binsi  = self.segment_bin_range(start, end, 0, self._bins[0], vox_size[0])
        for bini in binsi:
            # Restrict the line to a segment contained within this 2D slice
            si, ei = max(s0, bini*vox_size[0]), min(e0, (bini+1)*vox_size[0])
            ssi, sei = self.segment_end_points(si, ei, 0, start, grads)

            # Find the range of bins in the second axis, loop over them
            binsj = self.segment_bin_range(ssi, sei, 1, self._bins[1], vox_size[1])
            for binj in binsj:
                # Restrict the line to a segment contained within this 1D slice
                sj, ej = max(ssi[1], binj*vox_size[1]), min(sei[1], (binj+1)*vox_size[1])
                ssij, seij = self.segment_end_points(sj, ej, 1, start, grads)

                # Find the range of bins in the final axis, loop over them
                binsk = self.segment_bin_range(ssij, seij, 2, self._bins[2], vox_size[2])
                for bink in binsk:
                    # Restrict the line to a segment contained within this voxel
                    sk, ek = max(ssij[2], bink*vox_size[2]), min(seij[2], (bink+1)*vox_size[2])
                    ssijk, seijk = self.segment_end_points(sk, ek, 2, start, grads)

                    # Get the length of the segment inside the voxel
                    seg_len = np.linalg.norm(seijk-ssijk)

                    # Record bin IDs as voxel coordinates and length
                    idx = voxel_set.pos_to_id([bini, binj, bink])
                    voxel_set.add(idx, seg_len)

    @staticmethod
    def segment_end_points(si: nb.float64,
                           ei: nb.float64,
                           ai: nb.int64,
                           start: nb.int64,
                           grads: nb.float64[:]) -> (nb.float64[:], nb.float64[:]):
        '''
        Given a line start point and its gradients, find
        the coordinates of the start en end points of a segment
        of the line which starts and ends at given positions
        along one specific axis.

        Args:
            si (float): Start coordinate of segment along axis i
            ei (float): End coordinate of segment along axis i
            ai (int): Axis id (0,1 or 2)
            start (array): Track start point coordinates
            grads (array): Track gradients
        Returns:
            array: Segment start coordinates
            array: Segment end coordinates
        '''
        ss, se = np.empty(3), np.empty(3)
        xis, xie = (si - start[ai])/grads[ai], (ei - start[ai])/grads[ai]
        for a in range(3):
            ss[a] = start[a] + xis*grads[a]
            se[a] = start[a] + xie*grads[a]
            ss[a], se[a] = min(ss[a], se[a]), max(ss[a], se[a])

        return ss, se

    @staticmethod
    def segment_bin_range(ss: nb.float64[:],
                          se: nb.float64[:],
                          a: nb.int64,
                          n: nb.int64,
                          size: nb.float64) -> nb.int64[:]:
        '''
        Returns bin range of a segment along a certain axis

        Args:
            ss (array): Segment start point
            se (array): Segment end point
            a (int): Axis id (0,1 or 2)
            n (int): Number of bins along axis
            size (float): Bin size
        Returns:
            array: Range of bins
        '''
        s, e = min(ss[a], se[a]), max(ss[a], se[a])
        bins = np.arange(n)
        return bins[((bins+1)*size > s) & (bins*size < e)]

    @property
    def ranges(self):
        '''
        Returns the voxel ranges
        '''
        return self._ranges

    @property
    def bins(self):
        '''
        Returns the number of bins in each axis
        '''
        return self._bins

    @property
    def dimensions(self):
        '''
        Returns the dimensions of the image
        '''
        return self._ranges[:,1] - self._ranges[:,0]
