import argparse
import os
import time

import torch
import torch.nn as nn

from building_graph import NUM_DEP
from gcn import evaluate, load_graph
from model.factory import create_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("gcn", "rgcn"), default="gcn")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--dataset", default="hc3")
    parser.add_argument("--file", dest="test_file", required=True)
    parser.add_argument("--save", dest="do_save", action="store_true")
    parser.add_argument("--num_bases", type=int, default=8)
    parser.add_argument("--num_blocks", type=int)
    parser.add_argument("--add_reverse", type=int, choices=(0, 1), default=0)
    parser.add_argument("--use_labels", type=int, choices=(0, 1), default=1)
    parser.add_argument("--data_dir", "--graph_dir", dest="data_dir", default=os.environ.get(
        "PRDETECT_GRAPH_DIR", "/kaggle/working/graph_data"))
    parser.add_argument("--out_dir", default=os.environ.get("PRDETECT_OUT_DIR", "/kaggle/working/model"))
    parser.add_argument("--result_file")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args.model, num_bases=args.num_bases, num_blocks=args.num_blocks,
                         num_relations=2 * NUM_DEP).to(device)
    suffix = "gcn" if args.model == "gcn" else "rgcn"
    checkpoint = os.path.join(args.out_dir, f"{args.dataset}_{suffix}_model_{args.seed}.pth")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    start = time.perf_counter()
    metrics = evaluate(model, load_graph(os.path.join(args.data_dir, f"{args.test_file}.pkl")),
                       device, args.model, args.add_reverse, args.use_labels, nn.BCELoss())
    metrics["seconds"] = time.perf_counter() - start
    print(metrics)
    if args.do_save:
        with open("test_result.txt", "a", encoding="utf-8") as handle:
            handle.write(
                f"{args.test_file}\tmodel={args.model}\tbases={args.num_bases}\t"
                f"seed={args.seed}\tadd_reverse={args.add_reverse}\tuse_labels={args.use_labels}\t"
                f"acc={metrics['acc']}\tf1={metrics['f1']}\tauc={metrics['auc']}\t"
                f"parameters={sum(p.numel() for p in model.parameters())}\tseconds={metrics['seconds']}\n"
            )
    if args.result_file:
        with open(args.result_file, "w", encoding="utf-8") as handle:
            handle.write("acc,f1,auc,parameters,seconds\n")
            handle.write(
                f"{metrics['acc']},{metrics['f1']},{metrics['auc']},"
                f"{sum(p.numel() for p in model.parameters())},{metrics['seconds']}\n"
            )


if __name__ == "__main__":
    main()
