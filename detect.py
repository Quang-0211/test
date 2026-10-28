import argparse
import json
import os

import torch
from torch_geometric.data import Data

from building_graph import NUM_DEP, build_graph
from model.factory import create_model


def detect(text, dataset="hc3", seed=2024, model_name="gcn",
           out_dir=None, num_bases=8, add_reverse=0, use_labels=1):
    out_dir = out_dir or os.environ.get("PRDETECT_OUT_DIR", "/kaggle/working/model")
    record = json.dumps({"text": text, "label": ""})
    embeddings, edges, labels, edge_types = build_graph([record])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_name, num_bases=num_bases, num_relations=2 * NUM_DEP).to(device)
    suffix = "gcn" if model_name == "gcn" else "rgcn"
    model.load_state_dict(torch.load(
        os.path.join(out_dir, f"{dataset}_{suffix}_model_{seed}.pth"),
        map_location=device,
    ))
    kwargs = {"x": embeddings[0], "edge_index": edges[0]}
    if model_name == "rgcn":
        edge_type = edge_types[0]
        if not use_labels:
            edge_type = edge_type.clone().zero_()
        if add_reverse:
            kwargs["edge_index"] = torch.cat((edges[0], edges[0].flip(0)), dim=1)
            edge_type = torch.cat((edge_type, edge_type + NUM_DEP))
        keep = kwargs["edge_index"][0] != kwargs["edge_index"][1]
        kwargs["edge_index"] = kwargs["edge_index"][:, keep]
        kwargs["edge_type"] = edge_type[keep]
    model.eval()
    with torch.no_grad():
        probability = model(Data(**kwargs).to(device)).item()
    return probability, int(probability >= 0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--dataset", default="hc3")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--model", choices=("gcn", "rgcn"), default="gcn")
    parser.add_argument("--out_dir", default=os.environ.get("PRDETECT_OUT_DIR", "/kaggle/working/model"))
    args = parser.parse_args()
    print(detect(args.text, args.dataset, args.seed, args.model, args.out_dir))
