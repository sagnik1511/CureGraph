import yaml
import torch
import mlflow
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from src.models import GANN
from src.datasets import GNNDataset
from src.utils import calculate_scores
from torch_geometric.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

def train_single_epoch(loader, model, optim, loss_fn, epoch, device):
    epoch_loss = 0.0
    true_labels = []
    pred_labels = []
    for index, batch in enumerate(tqdm(loader)):
        batch.to(device)
        optim.zero_grad()
        op = model(batch.x,
                    batch.edge_attr,
                    batch.edge_index,
                    batch.batch)
        op = torch.squeeze(op)
        loss = loss_fn(op, batch.y.float())
        loss.backward()
        optim.step()
        epoch_loss += loss.item()
        true_labels += batch.y.cpu().detach().numpy().tolist()
        pred_labels += np.rint(op.cpu().detach().numpy()).tolist()
    calculate_scores(true_labels, pred_labels, epoch, "training")
    final_loss = round(epoch_loss / (index+1), 6)
    print(f"Training Loss : {final_loss}")
    mlflow.log_metric("training_loss", final_loss, step=epoch+1)

def test_single_epoch(loader, model, loss_fn, epoch, device):
    epoch_loss = 0.0
    true_labels = []
    pred_labels = []
    for index, batch in enumerate(tqdm(loader)):
        batch.to(device)
        op = model(batch.x,
                    batch.edge_attr,
                    batch.edge_index,
                    batch.batch)
        op = torch.squeeze(op)
        loss = loss_fn(op, batch.y.float())
        epoch_loss += loss.item()
        true_labels += batch.y.cpu().detach().numpy().tolist()
        pred_labels += np.rint(op.cpu().detach().numpy()).tolist()
    calculate_scores(true_labels, pred_labels, epoch, "testing")
    final_loss = round(epoch_loss / (index+1), 6)
    print(f"Testing Loss : {final_loss}")
    mlflow.log_metric("testing_loss", final_loss, step=epoch+1)
    return final_loss

def train(params):
    print("Process Initiated...")
    device = torch.device(f"cuda:"+params["training"]["gpu_node"] 
                            if torch.cuda.is_available() else "cpu")
    with mlflow.start_run() as run:
        for k, v in params.items():
            if isinstance(v, dict):
                mlflow.log_params(v)
            else:
                mlflow.log_param(k, v)
        print("Initializing Datasets...")
        train_dataset = GNNDataset(root=params["dataset"]["root"])
        test_dataset = GNNDataset(root=params["dataset"]["root"], test_data=True)
        feature_size = train_dataset[0]["x"].shape[1]
        params["model"]["model_edge_dim"] = train_dataset[0]["edge_attr"].shape[1]
        print("Datasets Generated...")
        train_loader = DataLoader(train_dataset,
                                    batch_size=params["dataset"]["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset,
                                    batch_size=params["dataset"]["batch_size"], shuffle=True)
        print("DataLoaders Generated...")
        print("Initializing Model...")
        model = GANN(feature_size=feature_size, model_params = params["model"])
        print("Model Generated...")
        print(model)
        model.to(device)
        print(f"Model Loaded on {device}\n")
        if params["training"]["loss_fn"] == "bce":
            loss_fn = nn.BCELoss()
        else:
            raise NotImplementedError
        if params["optimizer"]["name"] == "adam":
            optimizer = optim.Adam(model.parameters(),
                                     lr=params["optimizer"]["lr"],
                                      weight_decay=params["optimizer"]["weight_decay"])
        else:
            raise NotImplementedError

        best_loss = torch.inf
        num_epochs = params["training"]["num_epochs"]
        early_stop_counter = 0

        for epoch in range(num_epochs):
            if early_stop_counter == params["training"]["early_stop_count"]:
                print("Iteration Stopper due to Repeatative Degradation...")
                break
            else:
                print(f"Epoch {epoch + 1}: ")
                model.train()
                train_single_epoch(train_loader, model,
                                                 optimizer, loss_fn, epoch+1, device)
                current_loss = test_single_epoch(test_loader, model, loss_fn, epoch+1, device)
                if current_loss < best_loss:
                    best_loss = current_loss
                    mlflow.pytorch.log_state_dict(model.state_dict(), artifact_path="gann_best_weights")
                    early_stop_counter = 0
                    print("Model Weights Updated...")
                else:
                    early_stop_counter += 1
                print("\n")
    mlflow.end_run()
    print("Process Finished...")


if __name__ == '__main__':
    config_path = "config/default.yaml"
    with open(config_path, "r") as f:
        params = yaml.safe_load(f)
        f.close()
    train(params)
