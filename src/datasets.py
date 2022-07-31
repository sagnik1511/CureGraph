import os
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import deepchem as dc
from torch_geometric.data import Dataset, Data



class GNNDataset(Dataset):

    def __init__(self, root, test_data=False, transform=None, pre_transform=None, pre_filter=None):
        self.test_data = test_data
        self.name = "test" if self.test_data else "train"
        self.df = pd.read_csv(os.path.join(root, "raw", f"{self.name}.csv"))
        super().__init__(root, transform, pre_transform, pre_filter)
    
    @property
    def raw_file_names(self):
        return [f"{self.name}.csv"]
    
    @property
    def processed_file_names(self):
        return [f"{self.name}_{index+1}.pt" for index in range(len(self.df))]
    

    def download(self):
        pass

    def process(self):
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        for index, row in enumerate(tqdm(self.df.iterrows())):
            mol = featurizer.featurize(row[1]["smiles"])
            data = mol[0].to_pyg_graph()
            data.y = row[1]["HIV_active"]
            data.smiles = row[1]["smiles"]
            torch.save(data, 
                            os.path.join(self.processed_dir, f"{self.name}_{index+1}.pt"))

    def len(self):
        return len(self.df)

    def get(self, index):
        data = torch.load(os.path.join(self.processed_dir, f"{self.name}_{index+1}.pt"))
        return data


if __name__ == '__main__':
    root_dir = "data/"
    train_dataset = GNNDataset(root=root_dir)
    test_dataset = GNNDataset(root=root_dir, test_data=True)

    print(f"Length of training dataset : {len(train_dataset)}")
    print(f"Length of testing dataset : {len(test_dataset)}")
