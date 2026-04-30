from __future__ import annotations

import argparse
from pathlib import Path

from imgattck.experiments import check_tokens, invert_latent, optimize_latent, optimize_pixels, validate_native


def main() -> None:
    parser = argparse.ArgumentParser(prog="imgattck")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check-tokens", help="Validate that all target strings are single tokens.")
    check_parser.add_argument("config", type=Path)

    pixel_parser = subparsers.add_parser("optimize-pixels", help="Optimize pixels for target-token probability.")
    pixel_parser.add_argument("config", type=Path)

    native_parser = subparsers.add_parser("validate-native", help="Run official processor/model validation on an image.")
    native_parser.add_argument("config", type=Path)
    native_parser.add_argument("image", type=Path)

    latent_parser = subparsers.add_parser("optimize-latent", help="Optimize visual embeddings as an oracle.")
    latent_parser.add_argument("config", type=Path)

    invert_parser = subparsers.add_parser("invert-latent", help="Invert an optimized latent by pixel matching.")
    invert_parser.add_argument("config", type=Path)
    invert_parser.add_argument("latent", type=Path)

    args = parser.parse_args()
    if args.command == "check-tokens":
        run_dir = check_tokens(args.config)
    elif args.command == "optimize-pixels":
        run_dir = optimize_pixels(args.config)
    elif args.command == "validate-native":
        run_dir = validate_native(args.config, args.image)
    elif args.command == "optimize-latent":
        run_dir = optimize_latent(args.config)
    elif args.command == "invert-latent":
        run_dir = invert_latent(args.config, args.latent)
    else:
        parser.error(f"Unknown command: {args.command}")
        return
    print(run_dir)
