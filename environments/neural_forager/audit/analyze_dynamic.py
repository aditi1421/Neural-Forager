"""Analyze the separately frozen non-spiking smoothing follow-up."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from analyze import aggregate, contrast, holm


def analyze(prime):
    assert len(prime) == 16
    assert all(r.get("error") is None and r["is_completed"] and not r["is_truncated"] for r in prime)
    rows = [r for task in prime for r in task["dynamic"]["memory"]]
    backends = ("spiking", "rate", "rate_filter_8ms", "rate_filter_20ms")
    expected = {(s, n, t, b, c) for s in range(401, 409) for n in (31, 47)
                for t in ("spiking", "rate") for b in backends for c in ("clean", "current_noise")}
    keys = [(r["seed"], r["neural_seed"], r["trained_by"], r["backend"], r["condition"]) for r in rows]
    assert len(rows) == len(set(keys)) == 256 and set(keys) == expected
    comparisons = {}
    for backend in backends[2:]:
        selected = [{**r, "backend": "spiking" if r["backend"] == "spiking" else "rate",
                     "condition": r["condition"] + "/" + r["trained_by"]}
                    for r in rows if r["condition"] == "current_noise" and r["backend"] in ("spiking", backend)]
        comparisons[backend] = contrast(selected, "mse")
    holm(comparisons)
    for c in comparisons.values():
        c["filtered_rate_superiority_criterion"] = c["ci95"][0] > 0 and c["p_holm"] < 0.05
    return {"prime_tasks": 16, "independent_mazes": 8, "memory_assays": 256,
            "comparisons": comparisons,
            "means": {c: {b: aggregate([r for r in rows if r["condition"] == c and r["backend"] == b],
                                       ("accuracy", "mse", "seconds")) for b in backends}
                      for c in ("clean", "current_noise")},
            "by_weight_source": {t: {c: {b: aggregate([r for r in rows if r["condition"] == c and r["backend"] == b and r["trained_by"] == t],
                                                     ("accuracy", "mse")) for b in backends}
                                    for c in ("clean", "current_noise")} for t in ("spiking", "rate")}}


def tables(summary):
    lines = ["# Dynamic-rate follow-up", "", "Means pool both exact weight sources.", "",
             "| Condition | Receiver | MSE | Categorical accuracy |", "|---|---|---:|---:|"]
    for condition, means in summary["means"].items():
        for backend, m in means.items():
            lines.append(f'| {condition} | {backend} | {m["mse"]:.6f} | {100*m["accuracy"]:.3f}% |')
    lines += ["", "Positive MSE differences favor the non-spiking filter. Paired maze-bootstrap intervals.", "",
              "| Filter | Spiking minus filtered MSE | 95% interval | Holm p |", "|---|---:|---:|---:|"]
    for backend, c in summary["comparisons"].items():
        lines.append(f'| {backend} | {c["difference"]:.6f} | [{c["ci95"][0]:.6f}, {c["ci95"][1]:.6f}] | {c["p_holm"]:.5f} |')
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.source.read_bytes()
    raw = gzip.decompress(data) if args.source.suffix == ".gz" else data
    summary = analyze([json.loads(s) for s in raw.splitlines()])
    summary["source_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "dynamic-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "DYNAMIC_TABLES.md").write_text(tables(summary))
    if args.source.suffix != ".gz":
        (args.output / "dynamic-heldout.jsonl.gz").write_bytes(gzip.compress(raw, mtime=0))
    print(tables(summary))


if __name__ == "__main__":
    main()
