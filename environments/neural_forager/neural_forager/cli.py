import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from .config import ExperimentConfig
from .experiment import Experiment


def main():
    parser = argparse.ArgumentParser(description="Neural Forager: a spiking navigation experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Open the local live experiment")
    serve.add_argument("--port", type=int, default=8765)
    evaluate = commands.add_parser("evaluate", help="Paired, reproducible agent comparisons")
    evaluate.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103])
    evaluate.add_argument("--agents", nargs="+", choices=["neural", "frozen", "tabular", "random"],
                          default=["neural", "frozen", "tabular", "random"])
    evaluate.add_argument("--steps", type=int, default=1800)
    evaluate.add_argument("--size", type=int, default=9)
    evaluate.add_argument("--static", action="store_true")
    evaluate.add_argument("--output", type=Path, default=Path("results/comparison.json"))
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn
        uvicorn.run("neural_forager.server:app", host="127.0.0.1", port=args.port)
        return
    results = []
    for seed in args.seeds:
        for agent in args.agents:
            config = ExperimentConfig(seed=seed, agent=agent, steps=args.steps,
                                      size=args.size, changes=not args.static)
            with Experiment(config) as experiment:
                result = experiment.run()
            results.append(result)
            print(json.dumps(result["summary"]), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"protocol": "paired seeds; fresh network per maze; online learning; no pretrained weights",
                                              "runs": results}, indent=2))
    for agent in args.agents:
        food = [r["summary"]["food"] for r in results if r["summary"]["agent"] == agent]
        print(f"{agent:8s} food {mean(food):.2f} ± {stdev(food) if len(food) > 1 else 0:.2f} (SD, n={len(food)})")
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
