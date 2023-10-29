import numpy as np
from torch_geometric.data import Data
import torch

def convert_vector_to_graph(data, resolution = 35):
    """
        Converts subject vector to adjacency matrix then use it to create a graph.
        Input: vector containing elements in the lower-triangle of a matrix with dimension resolution x resolution (excluding the diagonal)
        Output: Data object describing a graph
    """

    data.reshape(1, resolution * (resolution - 1) / 2)
    # create adjacency matrix
    tri = np.zeros((resolution, resolution))
    # tri[np.triu_indices(35, 1)] = data
    tri[np.tril_indices(resolution, -1)] = data
    tri = tri + tri.T
    tri[np.diag_indices(resolution)] = 1

    edge_attr = torch.Tensor(tri).view(1225, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    counter = 0

    pos_edge_index = torch.zeros(2, resolution * resolution)
    for i in range(resolution):
        for j in range(resolution):
            pos_edge_index[:, counter] = torch.tensor([i, j])
            counter += 1

    x = torch.tensor(tri, dtype=torch.float)
    pos_edge_index = torch.tensor(pos_edge_index, dtype=torch.long)

    return Data(x=x, pos_edge_index=pos_edge_index, edge_attr=edge_attr)

def convert_list_of_vectors_to_graphs(dataset, resolution = 35):
    """
        Converts a list of subject vectors to graphs.
    """

    dataset_g = []

    for subj in range(dataset.shape[0]):
        dataset_g.append(convert_vector_to_graph(dataset[subj], resolution))

    return dataset_g