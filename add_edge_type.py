"""Add dependency relation labels to existing graph pickles without re-encoding."""
import argparse
import json
import os
import pickle

import spacy
import torch

from building_graph import dependency_relation_id


def augment(pickle_path, json_path, output_path=None):
    with open(pickle_path, "rb") as handle:
        graph = pickle.load(handle)
    with open(json_path, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    embeddings = graph["all_token_embeddings"]
    edge_indices = graph["all_edge_index"]
    if len(records) != len(embeddings):
        raise ValueError(f"{pickle_path}: record/graph count differs")
    nlp = spacy.load("en_core_web_sm")
    edge_types = []
    for index, record in enumerate(records):
        doc = nlp(record["text"])
        if len(doc) != embeddings[index].shape[0]:
            raise ValueError(f"{pickle_path}: token count differs at sample {index}")
        expected = torch.tensor(
            [[token.i, token.head.i] for token in doc], dtype=torch.long
        ).t().contiguous()
        if not torch.equal(expected, edge_indices[index]):
            raise ValueError(f"{pickle_path}: edge_index differs at sample {index}")
        edge_types.append(torch.tensor(
            [dependency_relation_id(token.dep_) for token in doc], dtype=torch.long
        ))
    graph["all_edge_type"] = edge_types
    destination = output_path or pickle_path
    with open(destination, "wb") as handle:
        pickle.dump(graph, handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle-dir", default=os.environ.get("PRDETECT_GRAPH_DIR", "/kaggle/working/graph_data"))
    parser.add_argument("--json-dir", default=os.environ.get("PRDETECT_INPUT_DIR", "/kaggle/input/PRDetect/original_text"))
    parser.add_argument("--files", nargs="+", required=True)
    args = parser.parse_args()
    for name in args.files:
        augment(
            os.path.join(args.pickle_dir, f"{name}.pkl"),
            os.path.join(args.json_dir, f"{name}.json"),
        )


if __name__ == "__main__":
    main()
