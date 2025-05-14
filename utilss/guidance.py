from torch_geometric.data import Dataset, Data, DataLoader
import torch
import random
from tqdm import tqdm
import torch.nn.functional as F
from pygda.models.gnn import GNN

def get_density(model, data):
    real_density, _ = model.predict(data)
    softmax_matrix = F.softmax(real_density, dim=1)
    result = softmax_matrix[:, 0] / softmax_matrix[:, 1]
    return torch.mean(result)

def train_domin_cls(source_data, target_data, device, source_test_data, target_test_data):
    x = torch.cat([source_data.x, target_data.x], dim=0)
    x_test = torch.cat([source_test_data.x, target_test_data.x], dim=0)
    edge_index_source = source_data.edge_index
    edge_index_target = target_data.edge_index + source_data.num_nodes
    edge_index_source_test = source_test_data.edge_index
    edge_index_target_test = target_test_data.edge_index + source_test_data.num_nodes
    edge_index = torch.cat([edge_index_source, edge_index_target], dim=1)
    edge_index_test = torch.cat([edge_index_source_test, edge_index_target_test], dim=1)
    y = torch.cat([
        torch.zeros(source_data.num_nodes, dtype=torch.long),
        torch.ones(target_data.num_nodes, dtype=torch.long)
    ], dim=0)
    y_test = torch.cat([
        torch.zeros(source_test_data.num_nodes, dtype=torch.long),
        torch.ones(target_test_data.num_nodes, dtype=torch.long)
    ], dim=0)
    data = Data(x=x, edge_index=edge_index, y=y).to(device)
    data_test = Data(x=x_test, edge_index=edge_index_test, y=y_test).to(device)
    num_features = data.x.size(1)
    model = GNN(in_dim=num_features, hid_dim=64, num_classes=2, device=device, epoch=50)
    model.fit(data, data_test)
    return model