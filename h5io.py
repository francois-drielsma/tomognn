import numpy as np
import h5py
from data_structures import VoxelSet, PixelSet

class EventWriter:
    '''
    Class which build an HDF5 file to store events which contain:
     - Voxelized track objects
     - Their projections
     - The node features associated with each projected point
     - The edge index
     - The edge labels (on/off)
    '''

    def __init__(self, file_name, bases):
        '''
        Initialize the output file

        Args:
            file_name: Name of the HDF5 file to output (will be overwritten)
            bases: Bases in which the voxels are projected
        '''
        # Store attributes
        self.file_name = file_name
        self.n_projs = len(bases)

        ref_dtype = h5py.special_dtype(ref=h5py.RegionReference)
        self.events_dtype  = [
            ('id', 'i8'),
            ('end_points_ref', ref_dtype),
            ('voxels_ref', ref_dtype),
            ('node_features_ref', ref_dtype),
            ('edge_index_ref', ref_dtype),
            ('edge_labels_ref', ref_dtype)
        ]
        for i in range(self.n_projs):
            self.events_dtype.append((f'pixels_{i}_ref', ref_dtype))
            self.events_dtype.append((f'node_features_{i}_ref', ref_dtype))

        # Initialize the output HDF5 file
        with h5py.File(file_name, 'w') as file:
            # Initialize the info dataset that stores characteristic of the dataset
            file.create_dataset('info', (0,), maxshape=(None,), dtype=None)
            file['info'].attrs['n_projs'] = self.n_projs

            # Initialize the events dataset
            file.create_dataset('events', (0,), maxshape=(None,), dtype=self.events_dtype)

            # Initialize the regular ndarray datasets
            file.create_dataset('end_points', (0,6), maxshape=(None,6), dtype=np.float64)
            file.create_dataset('voxels', (0,4), maxshape=(None,4), dtype=np.float64)
            for i in range(self.n_projs):
                file.create_dataset(f'pixels_{i}', (0,3), maxshape=(None,3), dtype=np.float64)
                file.create_dataset(f'node_features_{i}', (0,12), maxshape=(None,12), dtype=np.float64)
            file.create_dataset('edge_index', (0,4), maxshape=(None,4), dtype=np.int64)
            file.create_dataset('edge_labels', (0,), maxshape=(None,), dtype=np.bool_)

            # Store the projection axes as metadata
            for i in range(self.n_projs):
                for j in range(2):
                    file[f'pixels_{i}'].attrs[f'base'] = bases[i]

    def append(self, end_points, voxel_set, pixel_sets, node_features, edge_index, edge_labels):
        '''
        Append the HDF5 file with an event

        Args:
            end_points (array)         : End points of the lines
            voxel_set (dict)           : Sparse set of voxels
            pixel_sets (list(dict))    : List of sparse set of projected pixels
            node_features (list(array)): List of (N,12) arrays of node features
            edge_index (array)         : (E,2) array of edge index
            edge_labels (array)        : (E) array of edge labels
        '''
        with h5py.File(self.file_name, 'a') as file:
            # Append regular ndarray datasets, store region references
            ref_dict = {}
            end_points_array = np.hstack((end_points[0].reshape(-1,3), end_points[1].reshape(-1,3)))
            ref_dict['end_points_ref'] = self.store(end_points_array, file, 'end_points')
            ref_dict['voxels_ref'] = self.store(voxel_set.tensor, file, 'voxels')
            for i in range(self.n_projs):
                ref_dict[f'pixels_{i}_ref'] = self.store(pixel_sets[i].tensor, file, f'pixels_{i}')
                ref_dict[f'node_features_{i}_ref'] = self.store(node_features[i], file, f'node_features_{i}')
            ref_dict['edge_index_ref'] = self.store(edge_index, file, 'edge_index')
            ref_dict['edge_labels_ref'] = self.store(edge_labels, file, 'edge_labels')

            # Append event
            event_id  = len(file['events'])
            events_ds = file['events']
            events_ds.resize(event_id + 1, axis=0)

            event = np.empty(1, self.events_dtype)
            event['id'] = event_id
            for key, value in ref_dict.items():
                event[key] = value
            events_ds[event_id] = event

            # If not yet present, store the image attributes
            if 'bins' not in file['voxels'].attrs.keys():
                file['voxels'].attrs['ranges'] = voxel_set.ranges
                file['voxels'].attrs['bins'] = voxel_set.bins
            for i in range(self.n_projs):
                if 'bins' not in file[f'pixels_{i}'].attrs.keys():
                    file[f'pixels_{i}'].attrs['ranges'] = pixel_sets[i].ranges
                    file[f'pixels_{i}'].attrs['bins'] = pixel_sets[i].bins

    @staticmethod
    def store(array, file, name):
        '''
        Stores array in a specific dataset of an HDF5 file

        Args:
            array (array): Array to append the dataset with
            file (object): HDF5 file instance
            name (string): Name of the dataset
        Returns:
            object: HDF5 RegionReference of the newly stored array
        '''
        dataset = file[name]
        current_id = len(dataset)
        dataset.resize(current_id + len(array), axis=0)
        dataset[current_id:current_id + len(array)] = array
        return dataset.regionref[current_id:current_id + len(array)]


class EventReader:
    '''
    Class which reads an HDF5 file and builds the following objects:
     - Voxelized track objects
     - Their projections
     - The node features associated with each projected point
     - The edge index
     - The edge labels (on/off)
    '''

    def __init__(self, file_path):
        '''
        Initialize the file to read

        Args:
            file_path: Path to the HDF5 file to read
        '''
        # Store attributes
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as file:
            self.entries = len(file['events'])

    def __len__(self):
        '''
        Returns number of entries in the dataset
        '''
        return self.entries

    def __getitem__(self, idx):
        '''
        Returns a dictionary of the contents of an event

        Args:
            idx (int): Entry id
        '''
        with h5py.File(self.file_path, 'r') as file:
            return {'bases': self._bases(file),
                    'end_points': self._end_points(file, idx),
                    'voxel_set': self._voxel_set(file, idx),
                    'pixel_sets': self._pixel_sets(file, idx),
                    'node_features': self._node_features(file, idx),
                    'edge_index': self._edge_index(file, idx),
                    'edge_labels': self._edge_labels(file, idx)}

    def bases(self):
        '''
        Returns the projection bases the image was projected on

        Returns:
            list(array): List of projection bases
        '''
        with h5py.File(self.file_path, 'r') as file:
            return self._bases(file)

    def end_points(self, idx):
        '''
        Returns the voxel set of an entry

        Args:
            idx (int): Entry id
        Retuns:
            dict: Sparse set of voxels with their metadata
        '''
        with h5py.File(self.file_path, 'r') as file:
            return self._end_points(file, idx)

    def voxel_set(self, idx):
        '''
        Returns the voxel set of an entry

        Args:
            idx (int): Entry id
        Retuns:
            dict: Sparse set of voxels with their metadata
        '''
        with h5py.File(self.file_path, 'r') as file:
            return self._voxel_set(file, idx)

    def pixel_sets(self, idx):
        '''
        Returns the projected pixel sets of an entry

        Args:
            idx (int): Entry id
        Retuns:
            list(dict): List of sparse sets of pixels with their metadata
        '''
        with h5py.File(self.file_path, 'r') as file:
            return self._pixel_sets(file, idx)

    def graph(self, idx):
        '''
        Returns the graph information for an entry

        Args:
            idx (int): Entry id
        Retuns:
            array: Node features
            array: Edge index
            array: Edge labels
        '''
        with h5py.File(self.file_path, 'r') as file:
            return self._graph(file, idx)

    def _end_points(self, file, idx):
        tensor = self._tensor(file, 'end_points', idx)
        return tensor[:,:3], tensor[:,3:]

    def _voxel_set(self, file, idx):
        voxel_set = VoxelSet(file['voxels'].attrs['ranges'], file['voxels'].attrs['bins'])
        voxel_set.import_tensor(self._tensor(file, 'voxels', idx))
        return voxel_set

    def _pixel_sets(self, file, idx):
        n_projs    = file['info'].attrs['n_projs']
        pixel_sets = []
        for i in range(n_projs):
            proj_key  = f'pixels_{i}'
            pixel_set = PixelSet(file[proj_key].attrs['ranges'], file[proj_key].attrs['bins'])
            pixel_set.import_tensor(self._tensor(file, proj_key, idx))
            pixel_sets.append(pixel_set)

        return pixel_sets

    def _node_features(self, file, idx):
        n_projs    = file['info'].attrs['n_projs']
        node_features = []
        for i in range(n_projs):
            proj_key  = f'node_features_{i}'
            node_features.append(self._tensor(file, proj_key, idx))

        return node_features

    def _edge_index(self, file, idx):
        return self._tensor(file, 'edge_index', idx)

    def _edge_labels(self, file, idx):
        return self._tensor(file, 'edge_labels', idx)

    def _graph(self, file, idx):
        data_dict = {}
        for key in ['node_features', 'edge_index', 'edge_labels']:
            data_dict[key] = getattr(self, f'_{key}')(file, idx)

        return data_dict

    @staticmethod
    def _bases(file):
        n_projs = file['info'].attrs['n_projs']
        bases   = []
        for i in range(n_projs):
            bases.append(file[f'pixels_{i}'].attrs['base'])

        return bases

    @staticmethod
    def _tensor(file, key, idx):
        region = file['events'][idx][f'{key}_ref']
        return file[key][region]
