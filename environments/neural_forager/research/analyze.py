"""Analyze Prime's saved state, preserving paired maze-level uncertainty.

Usage: python research/analyze.py path/to/results.jsonl --output research
       python research/analyze.py research/heldout.json.gz --output /tmp/reanalysis
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

METHODS = ("selective", "ungated", "no_localization", "tabular")
SCENARIOS = ("stable", "reward", "route", "drift", "combined")


def aggregate(rows):
    return {
        "n": len(rows), "successes": sum(r["success"] for r in rows),
        **{k: float(np.mean([r[k] for r in rows])) for k in
           ("success", "memory_damage", "capped_latency", "pose_accuracy", "seconds", "abstentions")},
        "false_or_change_alarm_episodes": sum(bool(r["events"]) for r in rows),
        "first_diagnoses": dict(Counter(r["first_diagnosis"]["label"] if r["first_diagnosis"] else "none" for r in rows)),
        "channel_damage": {k: float(np.mean([r["channel_damage"][k] for r in rows]))
                           for k in ("geometry", "reward", "landmark")},
    }


def paired(rows, control, metric):
    indexed = {(r["seed"], r["neural_seed"], r["scenario"], r["method"]): r for r in rows}
    by_maze = {}
    for key, r in indexed.items():
        if key[-1] != "selective":
            continue
        other = indexed[(*key[:-1], control)]
        by_maze.setdefault(r["seed"], []).append(float(r[metric]) - float(other[metric]))
    values = np.array([np.mean(by_maze[s]) for s in sorted(by_maze)])
    rng = np.random.default_rng(20261001)
    boot = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return {"mean": float(values.mean()), "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "maze_differences": dict(zip(map(str, sorted(by_maze)), values.tolist()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.source.suffix == ".gz":
        payload = json.loads(gzip.decompress(args.source.read_bytes()))
        rows = payload["runs"]
        prime_count = payload["prime_rollouts"]
    else:
        prime = [json.loads(line) for line in args.source.read_text().splitlines()]
        assert len(prime) == 24, f"Expected 24 complete Prime rollouts, got {len(prime)}"
        assert all(r.get("error") is None and r["is_completed"] and not r["is_truncated"] for r in prime)
        rows = [episode for rollout in prime for episode in rollout["research"]]
        prime_count = len(prime)
    expected = {(s, n, c, m) for s in range(201, 213) for n in (31, 47) for c in SCENARIOS for m in METHODS}
    actual = [(r["seed"], r["neural_seed"], r["scenario"], r["method"]) for r in rows]
    assert len(rows) == 480 and len(set(actual)) == 480 and set(actual) == expected
    rows.sort(key=lambda r: (r["seed"], r["neural_seed"], r["scenario"], r["method"]))
    disrupted = [r for r in rows if r["scenario"] != "stable"]
    summary = {"episodes": len(rows), "prime_rollouts": prime_count, "independent_mazes": 12,
               "calibration_accuracy_min": min(r["calibration_accuracy"] for r in rows),
               "unapplied_intervention_episodes": sum(not r["intervention"]["applied"] for r in disrupted),
               "disrupted": {m: aggregate([r for r in disrupted if r["method"] == m]) for m in METHODS},
               "by_scenario": {s: {m: aggregate([r for r in rows if r["scenario"] == s and r["method"] == m])
                                   for m in METHODS} for s in SCENARIOS},
               "paired_contrasts": {m: {k: paired(disrupted, m, k) for k in
                                        ("memory_damage", "success", "capped_latency")} for m in METHODS[1:]},
               "failures": [{k: r[k] for k in ("seed", "neural_seed", "scenario", "method", "first_diagnosis", "pose_accuracy", "memory_damage")}
                            for r in rows if not r["success"]]}
    if summary["unapplied_intervention_episodes"]:
        applied = [r for r in disrupted if r["intervention"]["applied"]]
        summary["applied_only"] = {m: aggregate([r for r in applied if r["method"] == m]) for m in METHODS}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    payload = {"complete": True, "prime_rollouts": prime_count,
               "source_artifact_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(), "runs": rows}
    if args.source.suffix != ".gz":
        (args.output / "heldout.json.gz").write_bytes(gzip.compress(json.dumps(payload).encode(), mtime=0))
    lines = ["# Generated result tables", "", "Disrupted episodes only; 96 episodes per method.", "",
             "| Method | Recovery | Unchanged-memory damage | Capped moves | Pose accuracy |",
             "|---|---:|---:|---:|---:|"]
    for m, a in summary["disrupted"].items():
        lines.append(f'| {m} | {a["successes"]}/{a["n"]} | {100*a["memory_damage"]:.3f}% | {a["capped_latency"]:.2f} | {100*a["pose_accuracy"]:.1f}% |')
    lines += ["", "| Scenario | Method | Recovery | Damage | Capped moves | First diagnoses |", "|---|---|---:|---:|---:|---|"]
    for s, methods in summary["by_scenario"].items():
        for m, a in methods.items():
            lines.append(f'| {s} | {m} | {a["successes"]}/{a["n"]} | {100*a["memory_damage"]:.3f}% | {a["capped_latency"]:.2f} | {a["first_diagnoses"]} |')
    lines += ["", "Paired differences: selective minus control; percentile 95% cluster bootstrap over 12 mazes.", "",
              "| Control | Metric | Difference | 95% interval |", "|---|---|---:|---:|"]
    for m, metrics in summary["paired_contrasts"].items():
        for k, a in metrics.items():
            scale = 1 if k == "capped_latency" else 100
            unit = "moves" if scale == 1 else "percentage points"
            lo, hi = a["ci95"]
            lines.append(f'| {m} | {k} ({unit}) | {scale*a["mean"]:.3f} | [{scale*lo:.3f}, {scale*hi:.3f}] |')
    (args.output / "TABLES.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
