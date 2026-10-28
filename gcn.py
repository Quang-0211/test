import argparse
import os
import pickle
import time

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.data import Data

from building_graph import NUM_DEP
from model.factory import create_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("gcn", "rgcn"), default="gcn")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--dataset", default="hc3")
    parser.add_argument("--num_bases", type=int, default=8)
    parser.add_argument("--num_blocks", type=int)
    parser.add_argument("--add_reverse", type=int, choices=(0, 1), default=0)
    parser.add_argument("--use_labels", type=int, choices=(0, 1), default=1)
    parser.add_argument("--data_dir", "--graph_dir", dest="data_dir", default=os.environ.get(
        "PRDETECT_GRAPH_DIR", "/kaggle/working/graph_data"))
    parser.add_argument("--out_dir", default=os.environ.get("PRDETECT_OUT_DIR", "/kaggle/working/model"))
    parser.add_argument("--epochs", type=int, default=40)
    return parser.parse_args()


def graph_data(graph, index, model_name, add_reverse=0, use_labels=1):
    edge_index = graph["all_edge_index"][index]
    kwargs = {
        "x": graph["all_token_embeddings"][index],
        "edge_index": edge_index,
        "y": graph["y"][index],
    }
    if model_name == "rgcn":
        if "all_edge_type" not in graph or graph["all_edge_type"] is None:
            raise KeyError("R-GCN requires all_edge_type; run add_edge_type.py first")
        edge_type = graph["all_edge_type"][index].clone()
        if not use_labels:
            edge_type.zero_()
        if add_reverse:
            reverse = edge_index.flip(0)
            edge_index = torch.cat((edge_index, reverse), dim=1)
            edge_type = torch.cat((edge_type, edge_type + NUM_DEP))
        keep = edge_index[0] != edge_index[1]
        kwargs["edge_index"] = edge_index[:, keep]
        kwargs["edge_type"] = edge_type[keep]
        if kwargs["edge_type"].numel() and kwargs["edge_type"].max() >= 2 * NUM_DEP:
            raise ValueError("edge_type exceeds the configured relation vocabulary")
    return Data(**kwargs)


def load_graph(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def evaluate(model, graph, device, model_name, add_reverse, use_labels, criterion):
    model.eval()
    probabilities, labels = [], []
    loss_total = 0.0
    with torch.no_grad():
        for index in range(len(graph["y"])):
            data = graph_data(graph, index, model_name, add_reverse, use_labels).to(device)
            output = model(data)
            loss_total += criterion(output, data.y.float().view(-1, 1)).item()
            probabilities.append(output.item())
            labels.append(int(data.y.item()))
    predictions = [int(value >= 0.5) for value in probabilities]
    return {
        "loss": loss_total / len(labels),
        "acc": sum(a == b for a, b in zip(labels, predictions)) / len(labels),
        "f1": f1_score(labels, predictions, zero_division=0),
        "auc": roc_auc_score(labels, probabilities) if len(set(labels)) > 1 else float("nan"),
        "probabilities": probabilities,
        "labels": labels,
    }


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_graph = load_graph(os.path.join(args.data_dir, f"{args.dataset}_train.pkl"))
    val_graph = load_graph(os.path.join(args.data_dir, f"{args.dataset}_val.pkl"))
    model = create_model(
        args.model, num_bases=args.num_bases, num_blocks=args.num_blocks,
        num_relations=2 * NUM_DEP,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    best_acc = -1.0
    for epoch in range(args.epochs):
        model.train()
        for index in range(len(train_graph["y"])):
            data = graph_data(train_graph, index, args.model, args.add_reverse, args.use_labels).to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), data.y.float().view(-1, 1))
            loss.backward()
            optimizer.step()
        validation = evaluate(model, val_graph, device, args.model, args.add_reverse, args.use_labels, criterion)
        if validation["acc"] >= best_acc:
            best_acc = validation["acc"]
            os.makedirs(args.out_dir, exist_ok=True)
            suffix = "gcn" if args.model == "gcn" else "rgcn"
            torch.save(model.state_dict(), os.path.join(
                args.out_dir, f"{args.dataset}_{suffix}_model_{args.seed}.pth"
            ))
        print(f"epoch={epoch + 1} val_acc={validation['acc']:.6f}")


if __name__ == "__main__":
    train(parse_args())
