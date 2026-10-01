import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from .config import ResearchConfig
from .runner import ResearchRunner


def main():
    parser = argparse.ArgumentParser(description="Selective-plasticity research assay")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19])
    parser.add_argument("--neural-seeds", type=int, nargs="+", default=[31])
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parent
    fingerprint = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(root.glob("*.py")))).hexdigest()
    payload = {"complete": False, "source_sha256": fingerprint,
               "versions": {key: version(key) for key in ["nengo", "numpy", "scipy", "verifiers"]},
               "maze_seeds": args.seeds, "neural_seeds": args.neural_seeds,
               "expected_runs": len(args.seeds) * len(args.neural_seeds) * 20, "runs": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for neural_seed in args.neural_seeds:
            config = ResearchConfig(seed=seed, neural_seed=neural_seed, horizon=args.horizon)
            results = ResearchRunner(config).run()
            payload["runs"].extend(results)
            payload["complete"] = len(payload["runs"]) == payload["expected_runs"]
            temporary = args.output.with_suffix(".pending.json")
            temporary.write_text(json.dumps(payload, indent=2))
            temporary.replace(args.output)
            for row in results:
                print(json.dumps({key: row[key] for key in ["seed", "neural_seed", "scenario", "method", "success", "latency", "memory_damage", "pose_accuracy", "first_diagnosis"]}), flush=True)
    print(f"Saved {len(payload['runs'])} episodes to {args.output}", flush=True)


if __name__ == "__main__":
    main()
