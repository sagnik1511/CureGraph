import torch
import numpy as np
import deepchem as dc
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Draw import MolToImage
import warnings
warnings.filterwarnings("ignore")


featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)


def predict_from_smiles(smiles_string, model):
    mol_data = featurizer.featurize(smiles_string)[0]
    mol_data = mol_data.to_pyg_graph()

    x = torch.tensor(mol_data.x).float()
    edge_attributes = torch.tensor(mol_data.edge_attr).float()
    edge_index = torch.tensor(mol_data.edge_index).long()
    batch_index = torch.ones_like(mol_data.x[:, 0]).long()

    output = model(x=x, edge_attr=edge_attributes,
                    edge_index=edge_index, batch_index=batch_index)
    
    pred = np.rint(output.float().detach().numpy().ravel()[0])
    score = output.float().detach().numpy().ravel()[0] * 100.0
    prob = max(score, 100.0 - score)

    return pred, prob


def draw(smiles_string):
    return MolToImage(MolFromSmiles(smiles_string))


def validate_smiles_string(smile_string):
    flag = True
    try:
        assert MolFromSmiles(smile_string)
    except:
        flag = False

    return flag
