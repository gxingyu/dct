import torch
from torch_geometric.utils import subgraph

def get_subgraph(count, source_data):
    num_nodes = source_data.num_nodes
    subset_nodes = torch.randperm(num_nodes)[:count]
    remaining_nodes = torch.tensor([i for i in range(num_nodes) if i not in subset_nodes], dtype=torch.long)
    remaining_nodes = remaining_nodes.to(source_data.edge_index.device)
    subset_nodes = subset_nodes.to(source_data.edge_index.device)
    edge_index, edge_attr = subgraph(subset_nodes, source_data.edge_index, source_data.edge_attr)
    node_mapping = torch.zeros(num_nodes, dtype=torch.long).to(source_data.edge_index.device)
    node_mapping[subset_nodes] = torch.arange(len(subset_nodes), dtype=torch.long).to(source_data.edge_index.device)
    subgraph_edge_index = node_mapping[edge_index]
    subgraph_data = source_data.__class__()
    subgraph_data.x = source_data.x[subset_nodes]
    subgraph_data.y = source_data.y[subset_nodes]
    subgraph_data.edge_index = subgraph_edge_index
    subgraph_data.edge_attr = edge_attr
    subgraph_data.num_nodes = subset_nodes.size(0)
    subgraph_data.node_mapping = subset_nodes
    remaining_edge_index, remaining_edge_attr = subgraph(remaining_nodes, source_data.edge_index, source_data.edge_attr)
    remaining_node_mapping = torch.zeros(num_nodes, dtype=torch.long).to(source_data.edge_index.device)
    remaining_node_mapping[remaining_nodes] = torch.arange(len(remaining_nodes), dtype=torch.long).to(source_data.edge_index.device)
    remaining_subgraph_edge_index = remaining_node_mapping[remaining_edge_index]
    remaining_subgraph_data = source_data.__class__()
    remaining_subgraph_data.x = source_data.x[remaining_nodes]
    remaining_subgraph_data.y = source_data.y[remaining_nodes]
    remaining_subgraph_data.edge_index = remaining_subgraph_edge_index
    remaining_subgraph_data.edge_attr = remaining_edge_attr
    remaining_subgraph_data.num_nodes = remaining_nodes.size(0)
    remaining_subgraph_data.node_mapping = remaining_nodes
    return subgraph_data, remaining_subgraph_data

def update(source_data, x, edge_index, y, subgraph_data):
    source_data.x[subgraph_data.node_mapping] = x
    source_data.y[subgraph_data.node_mapping] = y
    edge_index_in_source = subgraph_data.node_mapping[edge_index]
    source_edge_index_set = set(map(tuple, source_data.edge_index.t().tolist()))
    new_edge_index_set = set(map(tuple, edge_index_in_source.t().tolist()))
    combined_edge_index_set = source_edge_index_set.union(new_edge_index_set)
    combined_edge_index = torch.tensor(list(combined_edge_index_set), dtype=torch.long).t()
    source_data.edge_index = combined_edge_index
    return source_data