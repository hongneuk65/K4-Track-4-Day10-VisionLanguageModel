"""Usage: python -m lab --help. No edits to the instructor tools."""
import argparse
import copy
import json
from pathlib import Path

from lab.runtime import write_json


def load_config(args):
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    for key in ("batch_size", "grad_accum", "max_steps"):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    if args.name:
        cfg["name"] = args.name
    if getattr(args, "rank", None) is not None:
        cfg["lora"]["r"] = args.rank
        cfg["lora"]["alpha"] = 2*args.rank
    return cfg


def score_file(args):
    from scripts.score import main as score_main
    from scripts.labcommon import read_jsonl
    expected = {r["image_id"] for r in read_jsonl(Path(args.data)/"splits"/f"{args.split}.jsonl")}
    rows = read_jsonl(args.pred)
    if len(rows) != len(expected) or {r["image_id"] for r in rows} != expected:
        raise ValueError("Scoring requires exactly one prediction for every image in the split")
    argv = ["--pred", args.pred, "--name", args.name, "--out", args.out,
            "--splits-dir", str(Path(args.data)/"splits"), "--split", args.split, "--coord", args.coord]
    if args.run:
        run = json.loads(Path(args.run).read_text(encoding="utf-8"))
        if run["completed_steps"] != run["config"]["max_steps"]:
            raise ValueError("Training is not complete")
        if args.split == "test" and run["completed_steps"] != 300:
            raise ValueError("Submission scoring requires the lab's 300 optimizer steps")
        if run["coord"] != args.coord:
            raise ValueError("Coordinate mismatch")
        for key in ("trainable_params", "peak_vram_gb", "train_minutes"):
            argv += ["--meta", f"{key}={run[key]}"]
        argv += ["--meta", f"train_fraction={run['train_fraction']}",
                 "--meta", f"seed={run['config']['seed']}"]
    else:
        inference_path = Path(args.pred).with_suffix(".run.json")
        if inference_path.exists():
            inference = json.loads(inference_path.read_text(encoding="utf-8"))
            argv += ["--meta", "trainable_params=0", "--meta", "train_minutes=0",
                     "--meta", f"peak_vram_gb={inference['peak_vram_gb']}",
                     "--meta", f"inference_minutes={inference['inference_minutes']}"]
    score_main(argv)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    audit = commands.add_parser("audit")
    audit.add_argument("--data", default="data")
    audit.add_argument("--out", default="report/data_audit.json")
    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--data", default="data")
    train.add_argument("--backup")
    train.add_argument("--name")
    train.add_argument("--batch-size", type=int)
    train.add_argument("--grad-accum", type=int)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--rank", type=int)
    train.add_argument("--fraction", type=float, default=1.0)
    train.add_argument("--coord", choices=["norm1000", "pixel"], default="norm1000")
    predict = commands.add_parser("predict")
    predict.add_argument("--data", default="data")
    predict.add_argument("--split", choices=["val", "test"], default="val")
    predict.add_argument("--out", required=True)
    predict.add_argument("--checkpoint")
    predict.add_argument("--batch-size", type=int, default=4)
    predict.add_argument("--max-new-tokens", type=int, default=256)
    predict.add_argument("--limit", type=int)
    predict.add_argument("--coord", choices=["norm1000", "pixel"], default="norm1000")
    score = commands.add_parser("score")
    score.add_argument("--data", default="data")
    score.add_argument("--pred", required=True)
    score.add_argument("--name", required=True)
    score.add_argument("--out", required=True)
    score.add_argument("--run")
    score.add_argument("--split", choices=["val", "test"], default="val")
    score.add_argument("--coord", choices=["norm1000", "pixel"], default="norm1000")
    plots = commands.add_parser("illustrate")
    plots.add_argument("--data", default="data")
    plots.add_argument("--pred")
    plots.add_argument("--out", default="report/figures/ground_truth")
    plots.add_argument("--split", choices=["train", "val", "test"], default="val")
    plots.add_argument("--coord", choices=["norm1000", "pixel"], default="norm1000")
    report = commands.add_parser("report")
    report.add_argument("--data", default="data")
    report.add_argument("--runs", default="work/runs")
    report.add_argument("--out", default="report/REPORT_DRAFT.md")
    commands.add_parser("scale-probe")
    args = parser.parse_args()
    if args.command == "doctor":
        from lab.runtime import environment
        print(json.dumps(environment(), indent=2))
    elif args.command == "audit":
        from lab.data import audit_data
        result = audit_data(args.data)
        write_json(args.out, result)
        print(json.dumps(result, indent=2))
    elif args.command == "train":
        from lab.runtime import train
        train(load_config(args), args.data, args.out, args.fraction, args.coord, args.backup)
    elif args.command == "predict":
        from lab.runtime import predict
        predict(args.data, args.split, args.out, args.checkpoint, args.coord,
                args.batch_size, args.max_new_tokens, args.limit)
    elif args.command == "score":
        score_file(args)
    elif args.command == "illustrate":
        from lab.artifacts import illustrate
        print(illustrate(args.data, args.pred, args.split, args.coord, args.out))
    elif args.command == "report":
        from lab.artifacts import report_draft
        print(report_draft(args.data, args.runs, out=args.out))
    elif args.command == "scale-probe":
        from lab.artifacts import scale_probe
        result = scale_probe()
        write_json("report/scale_probe.json", result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
