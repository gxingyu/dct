from model.dugda import DUGDA
from easydict import EasyDict as edict
import yaml
import argparse
import os.path as osp
import numpy as np
import torch
from pygda.datasets import CitationDataset, AirportDataset, BlogDataset
from torch_geometric.utils import degree
from pygda.metrics import eval_micro_f1, eval_macro_f1

parser = argparse.ArgumentParser()
#  Citation/ACMv9 DBLPv7 Citationv1
#  Airport/BRAZIL EUROPE USA
#  Blog/Blog1 Blog2
parser.add_argument('--nhid', type=int, default=64, help='hidden size')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--source', type=str, default='ACMv9', help='source domain data, DBLPv7/ACMv9/Citationv1')
parser.add_argument('--target', type=str, default='DBLPv7', help='target domain data, DBLPv7/ACMv9/Citationv1')
parser.add_argument('--config', type=str, default='ACMv9', help="Path of config file")
args = parser.parse_args()
# Load the configuration of the diffusion model
config_dir = f'./config/{args.config}.yaml'
config = edict(yaml.load(open(config_dir, 'r'), Loader=yaml.FullLoader))
# Load the source domain and target domain data
if args.source in {'DBLPv7', 'ACMv9', 'Citationv1'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', './data/Citation', args.source)
    source_dataset = CitationDataset(path, args.source)
if args.target in {'DBLPv7', 'ACMv9', 'Citationv1'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', './data/Citation', args.target)
    target_dataset = CitationDataset(path, args.target)
if args.source in {'BRAZIL', 'EUROPE', 'USA'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', './data/Airport', args.source)
    source_dataset = AirportDataset(path, args.source)
if args.target in {'BRAZIL', 'EUROPE', 'USA'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', './data/Airport', args.target)
    target_dataset = AirportDataset(path, args.target)
if args.source in {'Blog1', 'Blog2'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', './data/Blog', args.source)
    source_dataset = BlogDataset(path, args.source)
if args.target in {'Blog1', 'Blog2'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', './data/Blog', args.target)
    target_dataset = BlogDataset(path, args.target)
source_data = source_dataset[0].to(args.device)
target_data = target_dataset[0].to(args.device)
default_num_features = 241
# Check if source_data has the 'x' attribute; if not, construct features using OneHotDegree
if not hasattr(source_data, 'x') or source_data.x is None:
    # Calculate the degree of each node
    node_degrees = degree(source_data.edge_index[0], num_nodes=source_data.num_nodes).long()
    # Construct features using one-hot encoding
    source_data.x = torch.nn.functional.one_hot(node_degrees, num_classes=default_num_features).float().to(args.device)
# Check if target_data has the 'x' attribute; if not, construct features using OneHotDegree
if not hasattr(target_data, 'x') or target_data.x is None:
    # Calculate the degree of each node
    node_degrees = degree(target_data.edge_index[0], num_nodes=target_data.num_nodes).long()
    # Construct features using one-hot encoding
    target_data.x = torch.nn.functional.one_hot(node_degrees, num_classes=default_num_features).float().to(args.device)
num_features = source_data.x.size(1)
num_classes = len(np.unique(source_data.y.cpu().numpy()))
model = DUGDA(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device, config=config)
# Train the model
model.fit(source_data, target_data)
# Evaluate the performance
logits, labels = model.predict(target_data)
preds = logits.argmax(dim=1)
mi_f1 = eval_micro_f1(labels, preds)
ma_f1 = eval_macro_f1(labels, preds)
print('micro-f1: ' + str(mi_f1))
print('macro-f1: ' + str(ma_f1))
