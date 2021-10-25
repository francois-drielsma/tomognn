import h5py
import torch
import numpy as np
import numba as nb
from h5io import EventReader
from torch_geometric.data import Data, Dataset, DataLoader

class ToyDataLoader(DataLoader):
    '''
    This DataLoader is designed to return an event with node and edge feature data.
    - Node: pixel in a projection
    - Edge: potential connection between pixels
    '''
    def __init__(self, file_path, batch_size, num_workers=4, shuffle=True):
        """
        Instantiate the dataset

        Args:
            file_path (str) : Path to the HDF5 data file
            batch_size (int):
        """
        # Load file, store pointer to the node and edge features datasets
        dataset = ToyDataset(file_path)
        super().__init__(dataset = dataset,
                         batch_size = batch_size,
                         num_workers = num_workers,
                         shuffle = shuffle,
                         pin_memory = False)

class ToyDataset(Dataset):
    '''
    This Dataset is designed to return an event with node and edge feature data.
    - Node: pixel in a projection
    - Edge: potential connection between pixels
    '''
    def __init__(self, file_path):
        """
        Instantiate the dataset

        Args:
            file_path (str): Path to the HDF5 data file
        """
        # Load file, store pointer to the node and edge features datasets
        self.reader  = EventReader(file_path)
        self.entries = len(self.reader)

    def __len__(self):
        '''
        Returns number of entries in the dataset
        '''
        return self.entries

    def __getitem__(self, idx):
        '''
        Returns a torch geometric data object

        Args:
            idx (int): Entry id
        '''
        # Get the subset of node and edge features that correspond to the requested event ID
        graph = self.reader.graph(idx)
        return gnn_input(graph, idx)

def gnn_input(graph, idx=-1):
    '''
    Function which takes the graph information and converts
    it to a torch-geometric-friendly structure
    '''
    node_features, edge_index, edge_labels = _gnn_input(nb.typed.List(graph['node_features']), graph['edge_index'], graph['edge_labels'])

    return Data(x          = torch.Tensor(node_features),
                edge_index = torch.Tensor(edge_index).long().T,
                edge_label = torch.Tensor(edge_labels).long(),
#                     edge_attr = edge_features, # TODO: Should maybe include edge features ? Unsure if meaningfull
#                     y = node_labels, # TODO: Should include a charge target ? Unsure if useful
                num_nodes  = len(node_features),
                index      = idx)

@nb.njit
def _gnn_input(features: nb.types.List(nb.float64[:,:]), pairs, edge_labels, idx=-1):
    # Stack node features from the different projections
    sizes   = np.empty(len(features), dtype=np.int64)
    offsets = np.zeros(len(features)+1, dtype=np.int64)
    for i in range(len(features)):
        sizes[i]       = len(features[i])
        offsets[i+1:] += sizes[i]
    node_features = np.empty((np.sum(sizes), features[0].shape[1]), dtype=np.float32)
    for i, f in enumerate(features):
        node_features[offsets[i]:offsets[i+1]] = f

    # Get the edge index
    edge_index = np.empty((len(pairs), 2), dtype=np.int64)
    for i, e in enumerate(pairs):
        edge_index[i,0] = offsets[e[0]]+e[2]
        edge_index[i,1] = offsets[e[1]]+e[3]
    edge_index = np.vstack((edge_index, edge_index[:,::-1]))

    # Get the edge labels
    edge_labels = np.concatenate((edge_labels, edge_labels))

    return node_features, edge_index, edge_labels
