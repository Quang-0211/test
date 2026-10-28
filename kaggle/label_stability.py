import argparse
import json
import spacy


def labels(nlp, text):
    return [token.dep_ for token in nlp(text)]


def compare(nlp, before, after):
    left, right = labels(nlp, before), labels(nlp, after)
    matched = min(len(left), len(right))
    changed = sum(a != b for a, b in zip(left[:matched], right[:matched]))
    return {
        "matched_tokens": matched,
        "changed_labels": changed,
        "stability": 1.0 - changed / matched if matched else None,
        "unmatched_before": len(left) - matched,
        "unmatched_after": len(right) - matched,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    args = parser.parse_args()
    nlp = spacy.load("en_core_web_sm")
    with open(args.jsonl, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            print(json.dumps(compare(nlp, record["original"], record["perturbed"])))


if __name__ == "__main__":
    main()
