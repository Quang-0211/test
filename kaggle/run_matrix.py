import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError as error:
    raise SystemExit("Install PyYAML before running this script") from error


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--seeds", default="2021,2022,2023,2024,2025")
    parser.add_argument("--data-dir", default=os.environ.get("PRDETECT_DATA_DIR", "/kaggle/working/graph_data"))
    parser.add_argument("--out-dir", default=os.environ.get("PRDETECT_OUT_DIR", "/kaggle/working/model"))
    parser.add_argument("--results-dir", default="/kaggle/working/results")
    args = parser.parse_args()
    with open(Path(__file__).parent / "configs" / "experiments.yaml", encoding="utf-8") as handle:
        experiments = yaml.safe_load(handle)["experiments"]
    selected = args.only.split(",") if args.only else list(experiments)
    seeds = [int(value) for value in args.seeds.split(",")]
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    for code in selected:
        config = experiments[code]
        for seed in seeds:
            result_path = results_dir / f"{code}_{seed}.csv"
            if result_path.exists():
                continue
            train = [sys.executable, str(ROOT / "gcn.py"), "--model", config["model"],
                     "--seed", str(seed), "--dataset", "hc3", "--data_dir", args.data_dir,
                     "--out_dir", args.out_dir, "--num_bases", str(config["num_bases"]),
                     "--add_reverse", str(config["add_reverse"]), "--use_labels", str(config["use_labels"])]
            subprocess.run(train, cwd=ROOT, check=True)
            test = [sys.executable, str(ROOT / "test.py"), "--model", config["model"],
                    "--seed", str(seed), "--dataset", "hc3", "--file", "hc3_test",
                    "--data_dir", args.data_dir, "--out_dir", args.out_dir,
                    "--num_bases", str(config["num_bases"]), "--add_reverse", str(config["add_reverse"]),
                    "--use_labels", str(config["use_labels"]), "--save",
                    "--result_file", str(result_path)]
            subprocess.run(test, cwd=ROOT, check=True)
            metrics = result_path.read_text(encoding="utf-8").splitlines()
            result_path.write_text(
                "config,seed,model,add_reverse,use_labels,num_bases," + metrics[0] + "\n"
                f"{code},{seed},{config['model']},{config['add_reverse']},{config['use_labels']},{config['num_bases']},{metrics[1]}\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
