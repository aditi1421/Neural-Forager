"""Reproduce the spiking audit from a saved Prime results.jsonl[.gz]."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def contrast(rows, metric):
    paired = {}
    for row in rows:
        key = tuple(row.get(k) for k in ("seed", "neural_seed", "budget", "condition", "scenario"))
        pair = paired.setdefault(key, {})
        assert row["backend"] not in pair, f"Duplicate receiver in {key}"
        pair[row["backend"]] = float(row[metric])
    mazes = {}
    for key, pair in paired.items():
        assert set(pair) == {"spiking", "rate"}
        mazes.setdefault(key[0], []).append(pair["spiking"] - pair["rate"])
    values = np.array([np.mean(mazes[s]) for s in sorted(mazes)])
    rng = np.random.default_rng(20261002)
    samples = rng.choice(values, size=(20000, len(values)), replace=True).mean(axis=1)
    # Exact one-sided paired sign-flip test, including ties. Units are mazes.
    signs = 2 * ((np.arange(2 ** len(values))[:, None] >> np.arange(len(values))) & 1) - 1
    null_means = signs @ values / len(values)
    return {"difference": float(values.mean()), "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
            "p_spiking_greater": float(np.mean(null_means >= values.mean() - 1e-12)),
            "maze_differences": dict(zip(map(str, sorted(mazes)), values.tolist()))}


def aggregate(rows, metrics):
    return {"n": len(rows), **{m: float(np.mean([r[m] for r in rows])) for m in metrics}}


def holm(contrasts):
    ordered = sorted(contrasts.items(), key=lambda kv: kv[1]["p_spiking_greater"])
    previous = 0.0
    for i, (_, value) in enumerate(ordered):
        adjusted = min(1.0, (len(ordered) - i) * value["p_spiking_greater"])
        previous = max(previous, adjusted)
        value["p_holm"] = previous


def analyze(prime):
    assert len(prime) == 32
    assert all(r.get("error") is None and r["is_completed"] and not r["is_truncated"] for r in prime)
    tasks = [r["audit"] for r in prime]
    memory = [r for task in tasks for r in task["memory"]]
    nav = [r for task in tasks for r in task["navigation"]]
    mk = [(r["seed"], r["neural_seed"], r["budget"], r["trained_by"], r["backend"], r["condition"]) for r in memory]
    nk = [(r["seed"], r["neural_seed"], r["backend"], r["condition"], r["scenario"]) for r in nav]
    conditions = ("clean", "current_noise", "lesion_50", "short_read")
    backends = ("spiking", "rate")
    expected_m = {(s, n, b, t, r, c) for s in range(301, 317) for n in (31, 47) for b in (1, 12)
                  for t in backends for r in backends for c in conditions}
    expected_n = {(s, n, r, c, z) for s in range(301, 317) for n in (31, 47) for r in backends
                  for c in conditions[:3] for z in ("drift", "combined")}
    assert len(mk) == len(set(mk)) == 1024 and set(mk) == expected_m
    assert len(nk) == len(set(nk)) == 384 and set(nk) == expected_n
    own = [r for r in memory if r["backend"] == r["trained_by"]]
    primary = contrast([r for r in own if r["budget"] == 12], "accuracy")
    primary["passes_declared_benefit_criterion"] = primary["difference"] >= 0.01 and primary["ci95"][0] > 0
    by_condition, comparisons, transfers = {}, {}, {}
    for budget in (1, 12):
        for condition in conditions:
            key = f"{budget}/{condition}"
            selected = [r for r in own if r["budget"] == budget and r["condition"] == condition]
            by_condition[key] = {b: aggregate([r for r in selected if r["backend"] == b],
                ("accuracy", "geometry_accuracy", "reward_accuracy", "landmark_accuracy", "reward_site_detected", "mse", "seconds")) for b in backends}
            comparisons[key] = contrast(selected, "accuracy")
            comparisons[key]["analog_mse"] = contrast(selected, "mse")
            transfers[key] = {t: contrast([r for r in memory if r["budget"] == budget and r["condition"] == condition and r["trained_by"] == t], "accuracy") for t in backends}
    holm(comparisons)
    navigation = {c: {"means": {b: aggregate([r for r in nav if r["condition"] == c and r["backend"] == b],
                    ("success", "capped_latency", "memory_damage", "pose_accuracy", "seconds")) for b in backends},
                     "contrasts": {k: contrast([r for r in nav if r["condition"] == c], k)
                                   for k in ("success", "capped_latency", "memory_damage")}}
                  for c in conditions[:3]}
    calibration = [r for task in tasks for r in task["calibration"]]
    return {"prime_tasks": len(prime), "independent_mazes": 16, "memory_assays": len(memory),
            "navigation_episodes": len(nav), "primary": primary, "by_condition": by_condition,
            "condition_contrasts": comparisons, "same_weight_transfer": transfers, "navigation": navigation,
            "navigation_all": {b: aggregate([r for r in nav if r["backend"] == b], ("success", "capped_latency")) for b in backends},
            "unapplied_navigation_interventions": sum(not r["intervention"]["applied"] for r in nav),
            "minimum_calibration_accuracy": min(r["accuracy"] for r in calibration),
            "minimum_tabular_accuracy": min(t["tabular"]["accuracy"] for t in tasks),
            "resources": {str(b): {key: next(r for r in own if r["budget"] == b)["resources"][key]
                                   for key in ("neurons", "weight_scalars", "weight_bytes", "cache_bytes")} for b in (1, 12)},
            "tabular_value_bytes": tasks[0]["tabular"]["value_bytes"],
            "failures": [{k: r[k] for k in ("seed", "neural_seed", "backend", "condition", "scenario", "capped_latency", "pose_accuracy", "first_diagnosis")}
                         for r in nav if not r["success"]]}


def tables(summary):
    p = summary["primary"]
    lines = ["# Generated spiking audit results", "",
             f'Primary spiking-minus-rate accuracy: {100*p["difference"]:.4f} percentage points; '
             f'95% interval [{100*p["ci95"][0]:.4f}, {100*p["ci95"][1]:.4f}].', "",
             "| Neurons/cell | Condition | Spiking accuracy | Rate accuracy | Difference (pp) | Holm p (spiking > rate) |",
             "|---:|---|---:|---:|---:|---:|"]
    for key, means in summary["by_condition"].items():
        budget, condition = key.split("/")
        c = summary["condition_contrasts"][key]
        lines.append(f'| {budget} | {condition} | {100*means["spiking"]["accuracy"]:.3f}% | {100*means["rate"]["accuracy"]:.3f}% | {100*c["difference"]:.3f} | {c["p_holm"]:.4f} |')
    lines += ["", "| Condition | Backend | Recovery | Capped moves | Pose accuracy |", "|---|---|---:|---:|---:|"]
    for condition, results in summary["navigation"].items():
        for backend, m in results["means"].items():
            lines.append(f'| {condition} | {backend} | {round(m["success"]*m["n"])}/{m["n"]} | {m["capped_latency"]:.2f} | {100*m["pose_accuracy"]:.2f}% |')
    lines += ["", "Analog errors are secondary descriptive endpoints; these intervals are unadjusted.", "",
              "| Neurons/cell | Condition | Spiking MSE | Rate MSE | Difference, 95% interval |", "|---:|---|---:|---:|---:|"]
    for key, means in summary["by_condition"].items():
        budget, condition = key.split("/")
        c = summary["condition_contrasts"][key]["analog_mse"]
        lines.append(f'| {budget} | {condition} | {means["spiking"]["mse"]:.5f} | {means["rate"]["mse"]:.5f} | {c["difference"]:.5f} [{c["ci95"][0]:.5f}, {c["ci95"][1]:.5f}] |')
    lines += ["", "| Neurons/cell | Condition | Training source | Same-weight receiver difference (pp) |", "|---:|---|---|---:|"]
    for key, transfers in summary["same_weight_transfer"].items():
        budget, condition = key.split("/")
        for source, c in transfers.items():
            lines.append(f'| {budget} | {condition} | {source} | {100*c["difference"]:.4f} |')
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.source.read_bytes()
    raw = gzip.decompress(data) if args.source.suffix == ".gz" else data
    prime = [json.loads(line) for line in raw.splitlines()]
    summary = analyze(prime)
    summary["source_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "TABLES.md").write_text(tables(summary))
    if args.source.suffix != ".gz":
        (args.output / "prime-heldout.jsonl.gz").write_bytes(gzip.compress(raw, mtime=0))
    print(tables(summary))


if __name__ == "__main__":
    main()
