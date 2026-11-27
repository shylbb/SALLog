"""A single command surface for local preparation, API review and explanations."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
from .config import load_config
from .storage import dumps


def _api_flags(parser):
    parser.add_argument("--execute", action="store_true", help="Send API requests; otherwise preview without reading a key")
    parser.add_argument("--limit", type=int, default=0, help="New requests this invocation; 0 means all eligible")
    parser.add_argument("--token-budget", type=int, default=2000000, help="Cumulative conservative reservation per stage, not exact tokens or currency")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--retry-uncertain", action="store_true", help="Explicitly resend timeout/crash cases")


def main(argv=None):
    p = argparse.ArgumentParser(description="SALlog: LogCMA → calling → knowledge review → acceptance → explanation")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("embed", "train", "build-bank", "prepare", "run"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", required=True, type=Path)
        cmd.add_argument("--output", required=True, type=Path)
        if name in ("train", "build-bank", "prepare"):
            cmd.add_argument("--embeddings", required=True, type=Path)
        if name in ("build-bank", "prepare"):
            cmd.add_argument("--checkpoint", required=True, type=Path)
        if name == "prepare":
            cmd.add_argument("--bank", required=True, type=Path)
        if name == "run":
            _api_flags(cmd)
    for name in ("review", "finalize", "explain", "status"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--run", required=True, type=Path)
        if name in ("review", "explain"):
            _api_flags(cmd)
    args = p.parse_args(argv)
    try:
        if args.command in ("embed", "train", "build-bank", "prepare", "run"):
            cfg = load_config(args.config)
        if args.command == "embed":
            from .encoder import embed_templates
            result = {"embeddings": str(embed_templates(cfg, args.output))}
        elif args.command == "train":
            from .small_model import train
            result = {"checkpoint": str(train(cfg, args.embeddings, args.output))}
        elif args.command == "build-bank":
            from .knowledge import build_banks
            result = {"bank": str(build_banks(cfg, args.checkpoint, args.embeddings, args.output))}
        elif args.command == "prepare":
            from .pipeline import prepare
            result = {"run": str(prepare(cfg, args.checkpoint, args.embeddings, args.bank, args.output))}
        elif args.command in ("review", "explain"):
            flags = {"execute_api": args.execute, "limit": args.limit, "token_budget": args.token_budget,
                     "retry_failed": args.retry_failed, "retry_uncertain": args.retry_uncertain}
            if args.command == "review":
                from .pipeline import review, finalize
                result = review(args.run, **flags)
                if args.execute:
                    result["detection"] = finalize(args.run)
            else:
                from .explanation import explain
                result = explain(args.run, **flags)
        elif args.command == "finalize":
            from .pipeline import finalize
            result = finalize(args.run)
        elif args.command == "status":
            from .pipeline import load_run, api_usage
            from .llm import latest
            from collections import Counter
            directory, manifest, _ = load_run(args.run)
            result = {"targets": manifest["targets"], "population": manifest["population"],
                      "review": dict(Counter(r["status"] for r in latest(directory, "review").values())),
                      "explanation": dict(Counter(r["status"] for r in latest(directory, "explanation").values())),
                      "api_usage": api_usage(directory)}
        else:
            from .encoder import embed_templates
            from .small_model import train
            from .knowledge import build_banks
            from .pipeline import prepare, review, finalize
            from .explanation import explain
            args.output.mkdir(parents=True, exist_ok=False)
            embeddings = embed_templates(cfg, args.output / "embeddings")
            checkpoint = train(cfg, embeddings, args.output / "detector")
            bank = build_banks(cfg, checkpoint, embeddings, args.output / "bank")
            run = prepare(cfg, checkpoint, embeddings, bank, args.output / "review")
            review(run, execute_api=args.execute, limit=args.limit, token_budget=args.token_budget,
                   retry_failed=args.retry_failed, retry_uncertain=args.retry_uncertain)
            result = {"run": str(run), "detection": finalize(run),
                      "explanation": explain(run, execute_api=args.execute and bool(cfg["explanation"]["generate_missing"]),
                                             limit=args.limit, token_budget=args.token_budget)}
        print(dumps(result))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        p.exit(2, f"ERROR: {exc}\n")

