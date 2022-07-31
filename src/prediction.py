import torch
import numpy as np
import deepchem as dc
from rdkit.Chem.Draw import MolToImage
import warnings
warnings.filterwarnings("ignore")
featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)

def predict_one_record(data, model):
    x = torch.tensor(data.x).float()
    edge_attributes = torch.tensor(data.edge_attr).float()
    edge_index = torch.tensor(data.edge_index).long()
    batch_index = torch.ones_like(data.x[:, 0]).long()

    output = model(x=x, edge_attr=edge_attributes,
                    edge_index=edge_index, batch_index=batch_index)
    
    output = np.rint(output.detach().numpy()[0]).astype("int")[0]

    return output


def smiles2mol(smile_string):
    mol_data = featurizer.featurize(smile_string)[0]

    return mol_data.to_pyg_graph()

def predict_from_smiles(smile_string, model):
    data = smiles2mol(smile_string)
    output = predict_one_record(data, model)

    return output
