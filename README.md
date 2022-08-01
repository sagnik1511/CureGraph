<div align="center">
<h1>CureGraph</h1>
<h3>HIV Inhibition Detector using GraphNN</h3>
<img width="600px" src="https://github.com/sagnik1511/CureGraph/blob/main/assets/banner.png"><br>
<img src="https://github.com/sagnik1511/CureGraph/blob/main/assets/love.svg">
<img src="https://github.com/sagnik1511/CureGraph/blob/main/assets/python.svg">
<img src="https://github.com/sagnik1511/CureGraph/blob/main/assets/sci.svg"><br>
<img src="https://github.com/sagnik1511/CureGraph/blob/main/assets/gnn.svg"><br>
<img src="https://github.com/sagnik1511/CureGraph/blob/main/assets/pt.svg">
<img src="https://github.com/sagnik1511/CureGraph/blob/main/assets/st.svg">
</div>


<h2>About</h2>
<i>Current day, there are multiple pieces of research are happening on drug discovery. After the generation of GNN, researchers started to use Deep Learning which takes every compound's internal data, then uses graphs they try to minimize the losses. This repo tries to solve a similar problem where the molecules are annotated in the format of binary labels where they can inhibit HIV disease or not.</i><br>
<i>The raw dataset has been taken from <a href="https://moleculenet.org/">moleculenet.org</a>. The target feature of the dataset is not near uniformity. So, I tried the drop off some of the records so that they may reach a valuable state, also it contains a good number of records.</i><br>
<i>Finally it is split into train.csv and test.csv</i><br>
<i>Number of records in train dataset : 3848</i><br>
<i>Number of records in train dataset : 962</i><br>

<h2>Features of the Project</h2>
<i>1. Working with Graph Attention Neural Networks.</i><br>
<i>2. Easily identifies molecules expected behaviour to HIV disease.</i><br>
<i>3. Getting hands on towards medicinal AI.</i><br>
<i>4. Integrated with mlflow to track training results.</i><br>
<i>5. Using Graph data structures somewhere different than Competitive Programming.</i>

## Installation & Usage
1. Install Python on device. Use this [link](https://www.python.org/downloads/).
2. Install Anaconda on device. Use this [tutorial](https://docs.anaconda.com/anaconda/install/).
3. Install RDKit and it's components. Check [rdkit.org/docs/Install.html](https://www.rdkit.org/docs/Install.html). The RDKit modules installation was somehow not with pip. So, strictly use `conda`. 
4. Install PyTorch with cuda for faster training.
Version details are shared below.

    
        torch==1.12.0+cu116
        torchvision==0.13.0+cu116
5. Install PyG (PyTorch Geometric to prepare graph datasets). Follow this link : [pytorch-geometric.readthedocs.io/](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
6. Clone the repository. Run this command on terminal

        git clone https://github.com/sagnik1511/CureGraph.git
7. Go inside repository using `cd`.
   
        cd CureGraph
8. Run the streamlit app using this command

        streamlit run app.py

9. If you want to train the model, follow the procedures below
    
a) Update the config/default.yaml as per your need.

```yaml
dataset:
    root: "data/"
    batch_size: 128
model:
    model_embedding_size: 64
    model_attention_heads: 2
    model_layers: 4
    model_dropout_rate: 0.2
    model_top_k_ratio: 0.5
    model_top_k_every_n: 1
    model_dense_neurons: 256
optimizer:
    name: "adam"
    lr: 0.001
    weight_decay: 0.00001
    momentum: NA  # some of the optimizer uses momentum, some of them don't. Use NA in case there are no parameter like momentum
training:
    loss_fn: "bce"
    max_epochs: 100
    early_stop_count: 10
    gpu_node: "0"
````
You can add your new configuration by mimicing this format. Just make sure the file is a `yaml` file and it is stored inside `config` directory.

b) Run the training scripts as a module.

    python -m src.train

c) Fire mlflow server and track model training jobs using this command.

    mlflow ui

---

## Module Functionalities Achieved

- [x] Effective Web-platform UI.
- [x] Training through CPU / GPU.
- [x] Basic Loggings & Reports.
- [x] MLFlow Integration.
- [ ] Endpoint Service.
- [ ] W&B Integration.

<div align = "center">
<h3>If you get any errors while running the code, please make a PR.</h3>
<h1>Thanks for Visiting!!!</h1>
<h1>If you like the project, do ⭐</h1>
</div>

<div align = "center"><h1>Also follow me on <a href="https://github.com/sagnik1511">GitHub</a> , <a href="https://kaggle.com/sagnik1511">Kaggle</a> , <a href="https://in.linkedin.com/in/sagnik1511">LinkedIn</a></h1></div>
