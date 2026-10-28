import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def mean_std(values):
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("--output", default="summary.csv")
    args = parser.parse_args()
    groups = defaultdict(lambda: defaultdict(list))
    for path in Path(args.results_dir).glob("*.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for metric in ("acc", "f1", "auc"):
                    if row.get(metric):
                        groups[row["config"]][metric].append(float(row[metric]))
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["config", "metric", "mean", "std"])
        for config, metrics in sorted(groups.items()):
            for metric, values in sorted(metrics.items()):
                mean, std = mean_std(values)
                writer.writerow([config, metric, mean, std])


if __name__ == "__main__":
    main()
