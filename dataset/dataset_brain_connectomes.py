from torch_geometric.data import Dataset

class MultiResolutionBrainConnectomeDataset(Dataset):
    def __init__(self, source_data, target_data, resolution):
        super().__init__(root=None, transform=None, pre_transform=None, pre_filter=None)

        self.samples = []

        for idx, _ in enumerate(range(len(source_data))):
            # Data object from torch_geometric.data describing a graph
            source = source_data[idx]
            target = target_data[idx]

            # Node feature matrix x (which in this case is equal to the edge_attr parameter from Data object) representing the adjacency matrix
            source_x = source.x.view(35, 35)
            target_x = target.x.view(resolution, resolution)

            sample = {'source': source, 'target': target, 'target_x': target_x, 'source_x': source_x}

            self.samples.append(sample)

    def len(self):
        return len(self.samples)

    def get(self, idx):
        sample = self.samples[idx]
        return sample